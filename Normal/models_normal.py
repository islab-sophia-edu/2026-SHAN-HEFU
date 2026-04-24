import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed
import math

# =========================================================
# 1. Basic Components
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
        
        # [CRITICAL FIX] 适配 Y-Up 坐标系
        # 你的 Dataset 输出的是 Global Y-Up 法线。
        # 这里的 PE 必须建立 Y-Up 的空间感，否则模型无法学习。
        
        # X 轴 (右)
        x_now = torch.cos(lat_rad) * torch.cos(lon_rad)   
        
        # Y 轴 (上) -> 对应 Lat/Height
        y_now = torch.sin(lat_rad) 
        
        # Z 轴 (前) -> 对应 Depth
        # 这里使用 sin(lon) 配合负号或正号均可，只要保持右手系拓扑
        z_now = torch.cos(lat_rad) * torch.sin(lon_rad) 

        coords_3d = torch.stack([x_now, y_now, z_now], dim=-1)
        
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
        self.norm = nn.LayerNorm(out_channels) # Optional, strictly ReshapeUp usually doesn't have LN inside pixelshuffle flow in pure CNN, but keeping it if it helps stability.
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
        # Handle padding if dimensions don't match exactly due to odd sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# =========================================================
# 2. PanoNormal (Pure ViT, 3-Channel Input)
# =========================================================

class PanoNormal(nn.Module):
    def __init__(self, img_size=32, patch_size=32, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, geometric_bias=True, 
                 grid_height=16, output_size=(512, 1024), drop_path_rate=0.,
                 **kwargs):
        super().__init__()
        
        # 1. ViT Encoder
        self.view_embed = nn.Sequential()
        # [Fix] 确保 PatchEmbed 使用正确的 in_chans (3)
        self.view_embed.add_module('proj', PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
                  drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Output indices for U-Net skip connections
        if depth == 12: self.out_indices = [3, 5, 7, 11]
        elif depth == 24: self.out_indices = [5, 11, 17, 23]
        elif depth == 32: self.out_indices = [7, 15, 23, 31] # For Huge
        else: self.out_indices = [depth//4-1, depth//2-1, depth*3//4-1, depth-1]

        self.grid_h = grid_height
        self.grid_w = 2 * grid_height
        self.output_size = output_size
        
        # 2. Decoder (Pure U-Net)
        
        self.neck_conv = nn.Conv2d(embed_dim, 512, kernel_size=1) 
        
        self.skip1_up = ReshapeUp(embed_dim, 256, scale_factor=2) 
        self.skip2_up = ReshapeUp(embed_dim, 128, scale_factor=4)
        self.skip3_up = ReshapeUp(embed_dim, 64, scale_factor=8)
        
        # Up Blocks
        # Up1: Input(512) + Skip1(256) = 768 -> 256
        self.up1 = Up(in_channels=768, out_channels=256, bilinear=True)
        # Up2: Input(256) + Skip2(128) = 384 -> 128
        self.up2 = Up(in_channels=384, out_channels=128, bilinear=True)
        # Up3: Input(128) + Skip3(64) = 192 -> 64
        self.up3 = Up(in_channels=192, out_channels=64, bilinear=True)
        
        # Prediction Head
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1) # Output 3 channels (Normal)
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
        # views: (B, N, 3, H, W)  <-- Expecting 3 channels here
        B, N, C, H_p, W_p = views.shape 
        
        # 1. ViT Encoder
        x = views.view(B*N, C, H_p, W_p)
        x = self.view_embed(x) 
        x = x.view(B, N, -1)
        
        # Add Angle PE
        pos_embed = self.angle_pos_embed(angles)
        x = x + pos_embed
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                features.append(self.norm(x))
        
        # Helper to reshape tokens back to grid layout (B, Embed, Grid_H, Grid_W)
        def to_grid(t):
            t = t[:, 1:, :] # Remove CLS
            # Reshape: (B, N, Dim) -> (B, Dim, N) -> (B, Dim, Grid_H, Grid_W)
            return t.permute(0, 2, 1).view(B, -1, self.grid_h, self.grid_w)

        f3 = to_grid(features[0]) 
        f5 = to_grid(features[1]) 
        f7 = to_grid(features[2]) 
        f11 = to_grid(features[3]) 
        
        # 2. Decoder Flow
        x = self.neck_conv(f11) # (B, 512, H_g, W_g)
        
        # Stage 1
        s1 = self.skip1_up(f7)  
        x = self.up1(x, s1)     
        
        # Stage 2
        s2 = self.skip2_up(f5)  
        x = self.up2(x, s2)     

        # Stage 3
        s3 = self.skip3_up(f3)  
        x = self.up3(x, s3)     
        
        # Final Upsample to Full Pano Resolution
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        logits = self.head(x) # (B, 3, H_pano, W_pano)
        
        # Normalize for Normal Vectors (Crucial for Normal Estimation)
        logits = F.normalize(logits, p=2, dim=1) 

        # Reshape for Patch-based Loss Calculation (B, N, 3, PatchH, PatchW)
        # This matches the shape expected by CosineLoss in engine_normal.py
        patch_h = self.output_size[0] // self.grid_h
        patch_w = self.output_size[1] // self.grid_w
        
        # (B, 3, GH*PH, GW*PW) -> (B, 3, GH, PH, GW, PW)
        x = logits.view(B, 3, self.grid_h, patch_h, self.grid_w, patch_w)
        # Permute to (B, GH, GW, 3, PH, PW) -> (B, N, 3, PH, PW)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, N, 3, patch_h, patch_w) 
        
        return x

# [Fix] Updated Defaults to 3 Channels
def vit_base_patch16(**kwargs):
    kwargs.setdefault('patch_size', 32)
    kwargs.setdefault('in_chans', 3) 
    return PanoNormal(embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_large_patch16(**kwargs):
    kwargs.setdefault('patch_size', 32)
    kwargs.setdefault('in_chans', 3)
    return PanoNormal(embed_dim=1024, depth=24, num_heads=16, **kwargs)

def vit_huge_patch14(**kwargs):
    kwargs.setdefault('patch_size', 32)
    kwargs.setdefault('in_chans', 3)
    return PanoNormal(embed_dim=1280, depth=32, num_heads=16, **kwargs)