import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial
import math

class ViewEmbedder(nn.Module):
    """Embeds entire perspective views into tokens."""
    def __init__(self, in_chans=3, embed_dim=768, img_size=224):
        super().__init__()
        from timm.models.layers import PatchEmbed
        self.proj = PatchEmbed(
            img_size=img_size, patch_size=img_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
    def forward(self, x):
        return self.proj(x)

class AnglePositionalEncoding(nn.Module):
    """Spherical Position Embedding."""
    def __init__(self, d_model: int, num_fourier_features: int = 256, geometric_bias: bool = True):
        super().__init__()
        if d_model < 2 * num_fourier_features:
            num_fourier_features = d_model // 2
        self.num_fourier_features = num_fourier_features
        self.geometric_bias = geometric_bias
        self._init_weights()

        if 2 * self.num_fourier_features != d_model:
            self.output_proj = nn.Linear(2 * self.num_fourier_features, d_model)
        else:
            self.output_proj = nn.Identity()

    def _init_weights(self):
        if self.geometric_bias:
            l_max = int(math.sqrt(self.num_fourier_features))
            frequencies = []
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    if len(frequencies) >= self.num_fourier_features: break
                    freq = torch.tensor([
                        l * math.cos(m * math.pi / (l + 1)) if l > 0 else 1.0,
                        l * math.sin(m * math.pi / (l + 1)) if l > 0 else 0.0,
                        l * 0.5 if l > 0 else 0.0
                    ], dtype=torch.float32)
                    frequencies.append(freq)
                if len(frequencies) >= self.num_fourier_features: break
            while len(frequencies) < self.num_fourier_features:
                frequencies.append(torch.randn(3))
            fourier_weights = torch.stack(frequencies[:self.num_fourier_features])
        else:
            fourier_weights = torch.randn(self.num_fourier_features, 3)
        
        self.register_parameter('fourier_weights', nn.Parameter(fourier_weights))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        lon_rad, lat_rad = torch.deg2rad(angles[..., 0]), torch.deg2rad(angles[..., 1])
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * -torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        coords_3d = torch.stack([x, y, z], dim=-1)
        p_k = torch.matmul(coords_3d, self.fourier_weights.T)
        fourier_features = torch.cat([torch.cos(p_k), torch.sin(p_k)], dim=-1)
        return self.output_proj(fourier_features)

class PanoViTClassifier(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=True, geometric_bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim

        self.view_embed = ViewEmbedder(in_chans=in_chans, embed_dim=embed_dim, img_size=img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, 
                  proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # [P0 Fix] MLP Head (D -> 2D -> C) + Dropout
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, num_classes)
            )
        else:
            self.head = nn.Identity()
            
        self.initialize_weights()

    # [P0 Fix] Expand head method
    def expand_head_to_4d(self):
        device = next(self.parameters()).device
        print(f"Expanding Head from 2D ({self.embed_dim*2}) to 4D ({self.embed_dim*4})...")
        
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.LayerNorm(self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.num_classes)
        ).to(device)
        
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # 被不小心删掉的权重初始化方法，现已加回
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'cls_token', 'angle_pos_embed.fourier_weights'}

    def forward_features(self, views, angles):
        B, N, C, H, W = views.shape
        x = self.view_embed(views.view(B*N, C, H, W)).view(B, N, -1)
        pos_embed = self.angle_pos_embed(angles)
        x = x + pos_embed
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.global_pool:
            return x[:, 1:].mean(dim=1) 
        return x[:, 0]

    def forward(self, views, angles):
        x = self.forward_features(views, angles)
        x = self.head(x)
        return x

def vit_base_patch16(**kwargs):
    return PanoViTClassifier(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_large_patch16(**kwargs):
    return PanoViTClassifier(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_huge_patch14(**kwargs):
    return PanoViTClassifier(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)