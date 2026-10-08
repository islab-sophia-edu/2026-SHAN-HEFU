import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, Mlp, DropPath


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
    Spherical position embedding used by the current non-GCTT PanoMAE baseline.

    angles:
        [B, N, 2], in degrees, ordered as [lon, lat]

    Mapping:
        lon, lat -> 3D unit vector on S^2:
            x = cos(lat) * cos(lon)
            y = -cos(lat) * sin(lon)
            z = sin(lat)

    Then apply Random Fourier Features and an MLP.
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


class SphericalAttentionBias(nn.Module):
    """
    ALiBi-style angular distance attention bias.

    For token directions p_i and p_j on S^2:
        bias_h(i, j) = -slope_h * arccos(p_i · p_j)

    This is the non-GCTT position-bias branch:
    - no gauge_angles
    - no local tangent-frame gauge
    - no oriented SPE
    """
    def __init__(self, num_heads: int, init_max_slope: float = 1.0):
        super().__init__()
        self.num_heads = num_heads

        slopes = torch.tensor(
            [init_max_slope / (2 ** i) for i in range(num_heads)],
            dtype=torch.float32,
        )
        self.log_slopes = nn.Parameter(torch.log(slopes))

    @staticmethod
    def angles_to_unit_vec(angles_deg: torch.Tensor) -> torch.Tensor:
        lon = torch.deg2rad(angles_deg[..., 0])
        lat = torch.deg2rad(angles_deg[..., 1])

        cos_lat = torch.cos(lat)
        x = cos_lat * torch.cos(lon)
        y = -cos_lat * torch.sin(lon)
        z = torch.sin(lat)

        return torch.stack([x, y, z], dim=-1)

    def forward(self, angles: torch.Tensor, include_cls: bool = True) -> torch.Tensor:
        """
        Args:
            angles: [B, N, 2], degrees, [lon, lat]
            include_cls: prepend zero row/col for CLS token

        Returns:
            bias: [B, H, N(+1), N(+1)]
        """
        p = self.angles_to_unit_vec(angles)  # [B, N, 3]

        cos_sim = torch.einsum("bnd,bmd->bnm", p, p).clamp(
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )
        ang_dist = torch.arccos(cos_sim)  # [B, N, N]

        slopes = torch.exp(self.log_slopes)  # [H]
        bias = -slopes.view(1, -1, 1, 1) * ang_dist.unsqueeze(1)  # [B, H, N, N]

        if include_cls:
            B, H, N, _ = bias.shape
            bias_padded = torch.zeros(
                B,
                H,
                N + 1,
                N + 1,
                device=bias.device,
                dtype=bias.dtype,
            )
            bias_padded[:, :, 1:, 1:] = bias
            return bias_padded

        return bias


class AttentionWithBias(nn.Module):
    """
    Multi-head self-attention with external additive attention bias.

    attn_bias:
        [B, H, N, N] or broadcastable to that shape
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class BlockWithBias(nn.Module):
    """
    ViT block that accepts spherical position attention bias.
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = AttentionWithBias(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x, attn_bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias=attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PanoViTClassifier(nn.Module):
    """
    PanoMAE downstream classifier, non-GCTT + spherical position bias.

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
        use_angular_bias=True,
        angular_bias_init_slope=1.0,
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

        self.use_angular_bias = use_angular_bias
        if self.use_angular_bias:
            self.enc_ang_bias = SphericalAttentionBias(
                num_heads=num_heads,
                init_max_slope=angular_bias_init_slope,
            )
        else:
            self.enc_ang_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            if self.use_angular_bias:
                blk = BlockWithBias(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
            else:
                blk = Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
            self.blocks.append(blk)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Same initialization policy as current PanoMAE baseline.
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

        if self.use_angular_bias:
            no_decay.add("enc_ang_bias.log_slopes")

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

        attn_bias = None
        if self.use_angular_bias:
            attn_bias = self.enc_ang_bias(angles, include_cls=True)

        for blk in self.blocks:
            if self.use_angular_bias:
                x = blk(x, attn_bias=attn_bias)
            else:
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
