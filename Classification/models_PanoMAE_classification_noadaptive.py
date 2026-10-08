from functools import partial
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, Mlp, DropPath


class ViewEmbedder(nn.Module):
    """Embeds each tangent view into one ViT token."""
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


class OrientedSphericalPositionalEncoding(nn.Module):
    """
    GCTT positional encoding.

    Legacy:
        SPE(theta_i, phi_i) encodes only the patch center p_i on S^2.

    GCTT:
        SPE(theta_i, phi_i, psi_i) encodes the oriented local spherical frame
        F_i = [x_i(psi_i), y_i(psi_i), p_i].

    There is no standalone GE(psi). The local in-plane gauge is absorbed into
    the same 3D geometric state that defines the tangent token.
    """
    def __init__(
        self,
        d_model: int,
        num_fourier_features: int = 256,
        geometric_bias: bool = True,
        max_frequency: float = 4.0,
        use_oriented_frame: bool = False,
    ):
        super().__init__()

        if d_model < 2 * num_fourier_features:
            num_fourier_features = d_model // 2

        self.num_fourier_features = num_fourier_features
        self.use_oriented_frame = use_oriented_frame
        self.input_dim = 9 if use_oriented_frame else 3

        if geometric_bias:
            fourier_weights = self._build_stratified_fourier_weights(
                num_fourier_features,
                input_dim=self.input_dim,
                max_frequency=max_frequency,
            )
        else:
            fourier_weights = torch.randn(num_fourier_features, self.input_dim) * 0.5

        self.fourier_weights = nn.Parameter(fourier_weights)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_fourier_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _build_stratified_fourier_weights(
        num_features: int,
        input_dim: int,
        max_frequency: float,
    ) -> torch.Tensor:
        dirs = torch.randn(num_features, input_dim)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)
        freqs = torch.linspace(0.5, max_frequency, num_features).unsqueeze(-1)
        return dirs * freqs

    @staticmethod
    def angles_to_unit_vec(angles_deg: torch.Tensor) -> torch.Tensor:
        theta = torch.deg2rad(angles_deg[..., 0])
        phi = torch.deg2rad(angles_deg[..., 1])

        cos_phi = torch.cos(phi)
        x = cos_phi * torch.cos(theta)
        y = -cos_phi * torch.sin(theta)
        z = torch.sin(phi)

        p = torch.stack([x, y, z], dim=-1)
        return torch.nn.functional.normalize(p, dim=-1)

    @staticmethod
    def angles_to_oriented_frame(
        angles_deg: torch.Tensor,
        gauge_angles_deg: torch.Tensor = None,
    ) -> torch.Tensor:
        theta = torch.deg2rad(angles_deg[..., 0])
        phi = torch.deg2rad(angles_deg[..., 1])

        if gauge_angles_deg is None:
            psi = torch.zeros_like(theta)
        else:
            if gauge_angles_deg.dim() == 3:
                gauge_angles_deg = gauge_angles_deg[..., 0]
            psi = torch.deg2rad(gauge_angles_deg).to(device=theta.device, dtype=theta.dtype)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        p = torch.stack(
            [
                cos_phi * cos_theta,
                -cos_phi * sin_theta,
                sin_phi,
            ],
            dim=-1,
        )

        x_axis = torch.stack(
            [
                -sin_theta,
                -cos_theta,
                torch.zeros_like(theta),
            ],
            dim=-1,
        )
        y_axis = torch.stack(
            [
                -sin_phi * cos_theta,
                sin_phi * sin_theta,
                cos_phi,
            ],
            dim=-1,
        )

        c = torch.cos(psi).unsqueeze(-1)
        s = torch.sin(psi).unsqueeze(-1)
        x_g = c * x_axis + s * y_axis
        y_g = -s * x_axis + c * y_axis

        p = torch.nn.functional.normalize(p, dim=-1)
        x_g = torch.nn.functional.normalize(x_g, dim=-1)
        y_g = torch.nn.functional.normalize(y_g, dim=-1)

        frame = torch.cat([x_g, y_g, p], dim=-1)
        frame = torch.nn.functional.normalize(frame, dim=-1)
        return frame

    def forward(
        self,
        angles: torch.Tensor,
        gauge_angles: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_oriented_frame:
            coords = self.angles_to_oriented_frame(angles, gauge_angles)
        else:
            coords = self.angles_to_unit_vec(angles)

        weights = self.fourier_weights.to(device=coords.device, dtype=coords.dtype)
        phase = 2.0 * math.pi * torch.matmul(coords, weights.T)
        fourier_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.mlp(fourier_features)


class SphericalAttentionBias(nn.Module):
    """
    ALiBi-style relative attention bias on S^2, optionally using oriented frames.

    Geometry:
        b_ij^h = -alpha_h * arccos(p_i^T p_j)

    GCTT oriented-frame term:
        b_ij^h += beta_h * 0.5 * (x_i^T x_j + y_i^T y_j)
    """
    def __init__(
        self,
        num_heads: int,
        init_max_slope: float = 1.0,
        use_gauge_bias: bool = False,
        gauge_bias_init: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_gauge_bias = use_gauge_bias

        slopes = torch.tensor(
            [init_max_slope / (2 ** i) for i in range(num_heads)],
            dtype=torch.float32,
        )
        self.log_slopes = nn.Parameter(torch.log(slopes))

        if use_gauge_bias:
            self.gauge_beta = nn.Parameter(
                torch.full((num_heads,), float(gauge_bias_init), dtype=torch.float32)
            )
        else:
            self.register_parameter("gauge_beta", None)

    @staticmethod
    def angles_to_unit_vec(angles_deg: torch.Tensor) -> torch.Tensor:
        return OrientedSphericalPositionalEncoding.angles_to_unit_vec(angles_deg)

    @staticmethod
    def angles_to_axes(
        angles_deg: torch.Tensor,
        gauge_angles_deg: torch.Tensor = None,
    ):
        theta = torch.deg2rad(angles_deg[..., 0])
        phi = torch.deg2rad(angles_deg[..., 1])

        if gauge_angles_deg is None:
            psi = torch.zeros_like(theta)
        else:
            if gauge_angles_deg.dim() == 3:
                gauge_angles_deg = gauge_angles_deg[..., 0]
            psi = torch.deg2rad(gauge_angles_deg).to(device=theta.device, dtype=theta.dtype)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        p = torch.stack(
            [
                cos_phi * cos_theta,
                -cos_phi * sin_theta,
                sin_phi,
            ],
            dim=-1,
        )
        x_axis = torch.stack(
            [
                -sin_theta,
                -cos_theta,
                torch.zeros_like(theta),
            ],
            dim=-1,
        )
        y_axis = torch.stack(
            [
                -sin_phi * cos_theta,
                sin_phi * sin_theta,
                cos_phi,
            ],
            dim=-1,
        )

        c = torch.cos(psi).unsqueeze(-1)
        s = torch.sin(psi).unsqueeze(-1)
        x_g = c * x_axis + s * y_axis
        y_g = -s * x_axis + c * y_axis

        p = torch.nn.functional.normalize(p, dim=-1)
        x_g = torch.nn.functional.normalize(x_g, dim=-1)
        y_g = torch.nn.functional.normalize(y_g, dim=-1)
        return p, x_g, y_g

    def forward(
        self,
        angles: torch.Tensor,
        gauge_angles: torch.Tensor = None,
        include_cls: bool = True,
    ) -> torch.Tensor:
        p = self.angles_to_unit_vec(angles)
        cos_sim = torch.einsum("bnd,bmd->bnm", p, p).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        ang_dist = torch.arccos(cos_sim)

        slopes = torch.exp(self.log_slopes)
        bias = -slopes.view(1, -1, 1, 1) * ang_dist.unsqueeze(1)

        if self.use_gauge_bias and gauge_angles is not None:
            _, x_g, y_g = self.angles_to_axes(angles, gauge_angles)
            x_sim = torch.einsum("bnd,bmd->bnm", x_g, x_g)
            y_sim = torch.einsum("bnd,bmd->bnm", y_g, y_g)
            frame_sim = 0.5 * (x_sim + y_sim)
            beta = self.gauge_beta.to(device=bias.device, dtype=bias.dtype)
            bias = bias + beta.view(1, -1, 1, 1) * frame_sim.unsqueeze(1)

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
    """Standard MHSA with additive attention bias [B, H, N, N]."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
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
    """ViT block that accepts external relative attention bias."""
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
    GCTT-aware panoramic ViT classifier for NoAdaptive fine-tuning.

    Input:
        views:        [B, N, C, H, W]
        angles:       [B, N, 2] in degrees
        gauge_angles: [B, N] or [B, N, 1] in degrees

    Token:
        z_i = PatchEmbed(P_i(theta_i, phi_i, psi_i))
              + SPE(theta_i, phi_i, psi_i)

    No standalone GE(psi_i) is used.
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
        use_gctt=True,
        gauge_num_frequencies=16,  # deprecated; kept for CLI/checkpoint compatibility
        gauge_scale_init=0.02,     # deprecated; kept for CLI/checkpoint compatibility
        use_angular_bias=True,
        angular_bias_init_slope=1.0,
        use_gauge_bias=True,
        gauge_bias_init=0.0,
    ):
        super().__init__()
        del gauge_num_frequencies, gauge_scale_init

        # NoAdaptive classification: there is no MAE mask sampler in this model.
        # The encoder consumes all tangent-view tokens for supervised fine-tuning.

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.use_gctt = use_gctt
        self.use_angular_bias = use_angular_bias
        self.use_gauge_bias = use_gauge_bias and use_gctt

        self.view_embed = ViewEmbedder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            img_size=img_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.angle_pos_embed = OrientedSphericalPositionalEncoding(
            d_model=embed_dim,
            geometric_bias=geometric_bias,
            use_oriented_frame=use_gctt,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_angular_bias:
            self.ang_bias = SphericalAttentionBias(
                num_heads=num_heads,
                init_max_slope=angular_bias_init_slope,
                use_gauge_bias=self.use_gauge_bias,
                gauge_bias_init=gauge_bias_init,
            )
            block_cls = BlockWithBias
        else:
            self.ang_bias = None
            block_cls = Block

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if use_angular_bias:
            self.blocks = nn.ModuleList(
                [
                    block_cls(
                        embed_dim,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    block_cls(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    @staticmethod
    def _ensure_gauge(gauge_angles, angles):
        if gauge_angles is None:
            return torch.zeros(
                angles.shape[0],
                angles.shape[1],
                1,
                device=angles.device,
                dtype=angles.dtype,
            )
        if gauge_angles.dim() == 2:
            gauge_angles = gauge_angles.unsqueeze(-1)
        return gauge_angles.to(device=angles.device, dtype=angles.dtype)

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # PatchEmbed conv needs explicit ViT-style initialization.
        w = self.view_embed.proj.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.view_embed.proj.proj.bias is not None:
            nn.init.zeros_(self.view_embed.proj.proj.bias)

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
            no_decay.add("ang_bias.log_slopes")
            if self.use_gauge_bias:
                no_decay.add("ang_bias.gauge_beta")

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "angle_pos_embed.mlp" in name and m.bias is not None:
                no_decay.add(f"{name}.bias")
        return no_decay

    def forward_features(self, views, angles, gauge_angles=None):
        B, N, C, H, W = views.shape

        if self.use_gctt:
            gauge_angles = self._ensure_gauge(gauge_angles, angles)
        else:
            gauge_angles = None

        x = self.view_embed(views.view(B * N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(
            angles,
            gauge_angles=gauge_angles if self.use_gctt else None,
        )

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)

        attn_bias = None
        if self.use_angular_bias:
            attn_bias = self.ang_bias(
                angles,
                gauge_angles=gauge_angles if self.use_gctt else None,
                include_cls=True,
            )

        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)

        x = self.norm(x)

        if self.global_pool:
            return x[:, 1:].mean(dim=1)
        return x[:, 0]

    def forward(self, views, angles, gauge_angles=None):
        x = self.forward_features(views, angles, gauge_angles=gauge_angles)
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
