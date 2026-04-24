import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed
import math

# =========================================================
# 1. 全局 CNN 细节提取器 (带全景环绕填充)
# =========================================================

class CircularPad(nn.Module):
    """
    专门针对全景图的填充：
    - 上下：做 Zero Padding (或者 Reflect)
    - 左右：做 Circular Padding (因为全景图左右是连着的)
    """
    def __init__(self, padding=1):
        super().__init__()
        self.pad = padding

    def forward(self, x):
        # x: (B, C, H, W)
        # Pad 顺序: (Left, Right, Top, Bottom)
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad, self.pad), mode='constant', value=0)
        return x

class GlobalDetailCapture(nn.Module):
    """
    输入: 拼合后的完整大图 (B, C, FullH, FullW)
    输出: 多尺度特征，边缘连续，无 Patch 伪影
    """
    def __init__(self, in_chans=3, base_dim=64):
        super().__init__()
        
        def make_layer(in_c, out_c):
            return nn.Sequential(
                CircularPad(1),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Layer 1: H/2, W/2
        self.layer1 = make_layer(in_chans, base_dim)
        # Layer 2: H/4, W/4
        self.layer2 = make_layer(base_dim, base_dim*2)
        # Layer 3: H/8, W/8
        self.layer3 = make_layer(base_dim*2, base_dim*4)

    def forward(self, x):
        f1 = self.layer1(x) # -> (B, 64, H/2, W/2)
        f2 = self.layer2(f1) # -> (B, 128, H/4, W/4)
        f3 = self.layer3(f2) # -> (B, 256, H/8, W/8)
        return [f1, f2, f3]

# =========================================================
# 2. 基础组件
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
                freq = torch.tensor([l * math.cos(m * math.pi / (l + 1)) if l > 0 else 1.0, l * math.sin(m * math.pi / (l + 1)) if l > 0 else 0.0, l * 0.5 if l > 0 else 0.0], dtype=torch.float32)
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
    def forward(self, x):
        return self.conv(x)

class ReshapeUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.pixel_shuffle(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2*channels, channels, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, vit_feat, cnn_feat):
        combined = torch.cat([vit_feat, cnn_feat], dim=1)
        combined = self.conv(combined)
        attn = self.attention(combined)
        return combined * attn

# =========================================================
# 3. 核心模型 PanoNormal
# =========================================================

class PanoNormal(nn.Module):
    def __init__(self, img_size=32, patch_size=32, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, geometric_bias=True, 
                 grid_height=16, output_size=(832, 1664), drop_path_rate=0.,
                 **kwargs):
        super().__init__()
        
        # -------------------------------------------------------------------------
        # [核心修复] 结构对齐 Checkpoint: view_embed.proj.proj
        # -------------------------------------------------------------------------
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

        if depth == 12: self.out_indices = [3, 5, 7, 11]
        elif depth == 24: self.out_indices = [5, 11, 17, 23]
        else: self.out_indices = [depth//4-1, depth//2-1, depth*3//4-1, depth-1]

        self.grid_h = grid_height
        self.grid_w = 2 * grid_height
        self.output_size = output_size
        
        # Global CNN Detail Capture
        self.global_detail = GlobalDetailCapture(in_chans=in_chans, base_dim=64)
        
        # Fusion Modules
        self.fusion1 = AttentionFusion(256)
        self.fusion2 = AttentionFusion(128)
        self.fusion3 = AttentionFusion(64)

        self.neck_conv = nn.Conv2d(embed_dim, 512, kernel_size=1) 
        
        self.skip1_up = ReshapeUp(embed_dim, 256, scale_factor=2) 
        self.skip2_up = ReshapeUp(embed_dim, 128, scale_factor=4)
        self.skip3_up = ReshapeUp(embed_dim, 64, scale_factor=8)
        
        # Decoder Upsampling (Channels calculated based on Fusion output)
        # Up1: Neck(512) + Fusion1(256) -> 768 in -> 256 out
        self.up1 = Up(in_channels=768, out_channels=256, bilinear=True)
        # Up2: Up1(256) + Fusion2(128) -> 384 in -> 128 out
        self.up2 = Up(in_channels=384, out_channels=128, bilinear=True)
        # Up3: Up2(128) + Fusion3(64) -> 192 in -> 64 out
        self.up3 = Up(in_channels=192, out_channels=64, bilinear=True)
        
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
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'cls_token', 'angle_pos_embed.fourier_weights'}

    def forward(self, views, angles):
        B, N, C, H_p, W_p = views.shape 
        
        # 1. Global CNN
        x_grid = views.view(B, self.grid_h, self.grid_w, C, H_p, W_p)
        x_full = x_grid.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_full = x_full.view(B, C, self.grid_h * H_p, self.grid_w * W_p)
        
        cnn_feats = self.global_detail(x_full)
        cnn_high = cnn_feats[0] # 64
        cnn_mid  = cnn_feats[1] # 128
        cnn_low  = cnn_feats[2] # 256

        # 2. ViT Encoder
        x = views.view(B*N, C, H_p, W_p)
        
        # [核心修复] 使用 view_embed
        x = self.view_embed(x) 
        
        x = x.view(B, N, -1)
        pos_embed = self.angle_pos_embed(angles)
        x = x + pos_embed
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                features.append(self.norm(x))
        
        def to_grid(t):
            t = t[:, 1:, :] 
            return t.permute(0, 2, 1).view(B, -1, self.grid_h, self.grid_w)

        f3 = to_grid(features[0]) 
        f5 = to_grid(features[1]) 
        f7 = to_grid(features[2]) 
        f11 = to_grid(features[3]) 
        
        x = self.neck_conv(f11) # -> 512
        
        # Stage 1
        s1_vit = self.skip1_up(f7) # -> 256
        s1_cnn = F.interpolate(cnn_low, size=s1_vit.shape[-2:], mode='bilinear', align_corners=False) # -> 256
        s1_combined = self.fusion1(s1_vit, s1_cnn) # -> 256 (AttentionFusion maintains dim)
        x = self.up1(x, s1_combined) # Up(512) + Cat(256) -> Conv(768->256)
        
        # Stage 2
        s2_vit = self.skip2_up(f5) # -> 128
        s2_cnn = F.interpolate(cnn_mid, size=s2_vit.shape[-2:], mode='bilinear', align_corners=False) # -> 128
        s2_combined = self.fusion2(s2_vit, s2_cnn) # -> 128
        x = self.up2(x, s2_combined) # Up(256) + Cat(128) -> Conv(384->128)

        # Stage 3
        s3_vit = self.skip3_up(f3) # -> 64
        s3_cnn = F.interpolate(cnn_high, size=s3_vit.shape[-2:], mode='bilinear', align_corners=False) # -> 64
        s3_combined = self.fusion3(s3_vit, s3_cnn) # -> 64
        x = self.up3(x, s3_combined) # Up(128) + Cat(64) -> Conv(192->64)
        
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        logits = self.head(x)
        logits = F.normalize(logits, p=2, dim=1) 

        patch_h = self.output_size[0] // self.grid_h
        patch_w = self.output_size[1] // self.grid_w
        x = logits.view(B, 3, self.grid_h, patch_h, self.grid_w, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, N, 3, patch_h, patch_w) 
        
        return x

def vit_base_patch16(**kwargs):
    kwargs.setdefault('patch_size', 32)
    return PanoNormal(embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_large_patch16(**kwargs):
    kwargs.setdefault('patch_size', 32)
    return PanoNormal(embed_dim=1024, depth=24, num_heads=16, **kwargs)

def vit_huge_patch14(**kwargs):
    kwargs.setdefault('patch_size', 32)
    return PanoNormal(embed_dim=1280, depth=32, num_heads=16, **kwargs)