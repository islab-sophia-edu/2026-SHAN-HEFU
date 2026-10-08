import inspect
import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class ViewEmbedder(nn.Module):
    """
    Embed each tangent / perspective view as one token.

    Input:
        x: [B*N, C, H, W]
    Output:
        tokens: [B*N, 1, D]
    """
    def __init__(self, in_chans=3, embed_dim=768, img_size=224):
        super().__init__()
        from timm.models.layers import PatchEmbed

        self.proj = PatchEmbed(
            img_size=img_size,
            patch_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

    def forward(self, x):
        return self.proj(x)


class AnglePositionalEncoding(nn.Module):
    """
    3D spherical positional encoding for the pure 3DPE ablation.

    angles:
        [B, N, 2], in degrees, ordered as [lon, lat]

    Mapping:
        lon, lat -> 3D unit vector on S^2:
            x = cos(lat) * cos(lon)
            y = -cos(lat) * sin(lon)
            z = sin(lat)

    Then apply learnable Fourier features and an MLP:
        PE(p) = MLP([sin(2*pi*pW^T), cos(2*pi*pW^T)])

    Important:
        geometric_bias only controls the initialization of Fourier weights.
        It is not PB / pairwise attention bias.
    """
    def __init__(
        self,
        d_model: int,
        num_fourier_features: int = 256,
        geometric_bias: bool = True,
        max_frequency: float = 4.0,
    ):
        super().__init__()

        if d_model < 2 * num_fourier_features:
            num_fourier_features = d_model // 2

        self.num_fourier_features = num_fourier_features

        if geometric_bias:
            fourier_weights = self._build_stratified_fourier_weights(
                num_fourier_features,
                max_frequency=max_frequency,
            )
            self.fourier_weights = nn.Parameter(fourier_weights)
        else:
            self.fourier_weights = nn.Parameter(
                torch.randn(num_fourier_features, 3) * 0.5
            )

        self.mlp = nn.Sequential(
            nn.Linear(2 * num_fourier_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _build_stratified_fourier_weights(
        num_features: int,
        max_frequency: float,
    ) -> torch.Tensor:
        dirs = torch.randn(num_features, 3)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)

        freqs = torch.linspace(0.5, max_frequency, num_features).unsqueeze(-1)
        return dirs * freqs

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        lon_rad = torch.deg2rad(angles[..., 0])
        lat_rad = torch.deg2rad(angles[..., 1])

        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = -torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        coords_3d = torch.stack([x, y, z], dim=-1)
        coords_3d = torch.nn.functional.normalize(coords_3d, dim=-1)

        phase = 2.0 * math.pi * torch.matmul(coords_3d, self.fourier_weights.T)
        fourier_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        return self.mlp(fourier_features)


def _make_standard_vit_block(
    embed_dim,
    num_heads,
    mlp_ratio,
    qkv_bias,
    drop_rate,
    attn_drop_rate,
    drop_path_rate,
    norm_layer,
):
    """
    Create a timm VisionTransformer Block with compatibility across timm versions.
    Some versions use `proj_drop`; older MAE/timm versions use `drop`.
    """
    sig = inspect.signature(Block)
    kwargs = {
        "dim": embed_dim,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": qkv_bias,
        "attn_drop": attn_drop_rate,
        "drop_path": drop_path_rate,
        "norm_layer": norm_layer,
    }

    if "proj_drop" in sig.parameters:
        kwargs["proj_drop"] = drop_rate
    elif "drop" in sig.parameters:
        kwargs["drop"] = drop_rate

    return Block(**kwargs)


class PanoViTClassifier(nn.Module):
    """
    PanoMAE downstream classifier: pure 3DPE ablation.

    This model intentionally removes all pairwise angular attention bias modules.

    It keeps only additive 3D spherical positional encoding:
        token = ViewEmbed(tangent_patch) + AnglePositionalEncoding(angles)

    Input:
        views:  [B, N, C, H, W]
        angles: [B, N, 2], degrees, [lon, lat]

    Output:
        logits: [B, num_classes]
    """
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=True,
        geometric_bias=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim

        self.view_embed = ViewEmbedder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            img_size=img_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.angle_pos_embed = AnglePositionalEncoding(
            d_model=embed_dim,
            geometric_bias=geometric_bias,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Pure 3DPE ablation: use standard ViT blocks, not bias-aware blocks.
        self.blocks = nn.ModuleList([
            _make_standard_vit_block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Same initialization policy as the 3DPE pretraining baseline.
        if hasattr(self.view_embed.proj, "proj"):
            w = self.view_embed.proj.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        no_decay = {
            "cls_token",
            "angle_pos_embed.fourier_weights",
        }

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "angle_pos_embed.mlp" in name:
                if m.bias is not None:
                    no_decay.add(f"{name}.bias")

        return no_decay

    def forward_features(self, views, angles):
        B, N, C, H, W = views.shape

        x = self.view_embed(views.view(B * N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles)

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
        return self.head(x)


def vit_base_patch16(**kwargs):
    return PanoViTClassifier(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large_patch16(**kwargs):
    return PanoViTClassifier(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_huge_patch14(**kwargs):
    return PanoViTClassifier(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
