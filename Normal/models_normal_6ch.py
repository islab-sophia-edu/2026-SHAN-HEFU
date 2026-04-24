import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed
import math

# =========================================================
# 1. 基础组件
# =========================================================

class AnglePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_fourier_features: int = 256, geometric_bias: bool = True):
        super().__init__()
        if d_model < 2 * num_fourier_features:
            num_fourier_features = d_model // 2
        self.num_fourier_features = num_fourier_features
        if geometric_bias:
            self._init_geometric_fourier_weights()
        else:
            self.register_parameter('fourier_weights', nn.Parameter(torch.randn(self.num_fourier_features, 3)))
        if 2 * self.num_fourier_features != d_model:
            self.output_proj = nn.Linear(2 * self.num_fourier_features, d_model)
        else:
            self.output_proj = nn.Identity()

    def _init_geometric_fourier_weights(self):
        l_max = int(math.sqrt(self.num_fourier_features))
        frequencies = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                if len(frequencies) >= self.num_fourier_features: break
                freq = torch.tensor([l * math.cos(m * math.pi / (l + 1)) if l > 0 else 1.0, 
                                   l * math.sin(m * math.pi / (l + 1)) if l > 0 else 0.0, 
                                   l * 0.5 if l > 0 else 0.0], dtype=torch.float32)
                frequencies.append(freq)
            if len(frequencies) >= self.num_fourier_features: break
        while len(frequencies) < self.num_fourier_features:
            frequencies.append(torch.randn(3))
        fourier_weights = torch.stack(frequencies[:self.num_fourier_features])
        self.register_parameter('fourier_weights', nn.Parameter(fourier_weights))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        lon_rad, lat_rad = torch.deg2rad(angles[..., 0]), torch.deg2rad(angles[..., 1])
        
        x_now = torch.cos(lat_rad) * torch.sin(lon_rad)
        y_now = torch.sin(lat_rad)                        # 现在的垂直轴 (Vertical)
        z_now = torch.cos(lat_rad) * torch.cos(lon_rad)   # 现在的深度轴
        
        x_for_weight = z_now
        y_for_weight = -1.0 * x_now
        z_for_weight = y_now
        coords_3d = torch.stack([x_for_weight, y_for_weight, z_for_weight], dim=-1)
        
        p_k = torch.matmul(coords_3d, self.fourier_weights.T)
        features = torch.cat([torch.cos(p_k), torch.sin(p_k)], dim=-1)
        return self.output_proj(features)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ReshapeUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
    def forward(self, x):
        return self.pixel_shuffle(self.conv1x1(x))

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# =========================================================
# 2. PanoNormal 主模型
# =========================================================

class PanoNormal(nn.Module):
    def __init__(self, img_size=32, patch_size=32, in_chans=6, 
                 embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, geometric_bias=True, 
                 grid_height=16, output_size=(512, 1024), drop_path_rate=0.,
                 **kwargs):
        super().__init__()
        
        self.view_embed = nn.Sequential()
        self.view_embed.add_module('proj', PatchEmbed(img_size, patch_size, in_chans, embed_dim))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
                  drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.out_indices = [depth//4-1, depth//2-1, depth*3//4-1, depth-1]
        self.grid_h = grid_height
        self.grid_w = 2 * grid_height
        self.output_size = output_size
        
        # Decoder 结构
        self.neck_conv = nn.Conv2d(embed_dim, 512, kernel_size=1) 
        self.skip1_up = ReshapeUp(embed_dim, 256, scale_factor=2) 
        self.skip2_up = ReshapeUp(embed_dim, 128, scale_factor=4)
        self.skip3_up = ReshapeUp(embed_dim, 64, scale_factor=8)
        
        self.up1 = Up(in_channels=768, out_channels=256)
        self.up2 = Up(in_channels=384, out_channels=128)
        self.up3 = Up(in_channels=192, out_channels=64)
        
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # [FIXED] 补齐缺失的方法
    def no_weight_decay(self):
        return {'cls_token', 'angle_pos_embed.fourier_weights'}

    def forward(self, views, angles):
        B, N, C, H_p, W_p = views.shape 
        
        # 1. Encoder
        x = self.view_embed(views.view(B*N, C, H_p, W_p))
        if x.ndim == 4: x = x.flatten(2).transpose(1, 2)
        x = x.mean(dim=1).view(B, N, -1)

        x = x + self.angle_pos_embed(angles)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                features.append(self.norm(x))
        
        def to_grid(t):
            t = t[:, 1:, :].transpose(1, 2)
            return t.view(B, -1, self.grid_h, self.grid_w)

        f_list = [to_grid(f) for f in features]
        
        # 2. Decoder
        x = self.neck_conv(f_list[3]) 
        x = self.up1(x, self.skip1_up(f_list[2]))     
        x = self.up2(x, self.skip2_up(f_list[1]))     
        x = self.up3(x, self.skip3_up(f_list[0]))     
        
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        logits = F.normalize(self.head(x), p=2, dim=1) 

        # 3. Reshape 为 Patch 格式以计算 Loss
        ph, pw = self.output_size[0] // self.grid_h, self.output_size[1] // self.grid_w 
        x = logits.view(B, 3, self.grid_h, ph, self.grid_w, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, N, 3, ph, pw) 
        return x

# Factories
def vit_base_patch16(**kwargs):
    kwargs.setdefault('patch_size', 32)
    kwargs.setdefault('in_chans', 6)
    return PanoNormal(embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_large_patch16(**kwargs):
    kwargs.setdefault('patch_size', 32)
    kwargs.setdefault('in_chans', 6)
    return PanoNormal(embed_dim=1024, depth=24, num_heads=16, **kwargs)

def vit_huge_patch14(**kwargs):
    kwargs.setdefault('patch_size', 32)
    kwargs.setdefault('in_chans', 6)
    return PanoNormal(embed_dim=1280, depth=32, num_heads=16, **kwargs)