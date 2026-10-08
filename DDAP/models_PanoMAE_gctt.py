from functools import partial
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, Mlp, DropPath


class ViewEmbedder(nn.Module):
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
        # For an SxS tangent view with patch_size=S, timm PatchEmbed returns [B, 1, D].
        return self.proj(x)


class OrientedSphericalPositionalEncoding(nn.Module):
    """
    Rotation-matrix / frame-based spherical positional encoding.

    Legacy spherical PE encodes only the patch center p_i on S^2:
        PE(p_i),  p_i = p(theta_i, phi_i).

    With GCTT enabled, the token geometry is not only a point on S^2. The tangent
    patch also has a local in-plane gauge psi_i. Therefore the encoded object is:
        (p_i, psi_i) in S^2 x S^1,

    or equivalently an oriented local spherical frame:
        F_i = [x_i(psi_i), y_i(psi_i), p_i] in SO(3).

    This module encodes SPE(theta_i, phi_i, psi_i) by applying learnable Fourier
    features to the full oriented frame entries. It deliberately removes the old
    standalone GE(psi_i) path: psi_i is no longer encoded as an independent
    periodic scalar feature. Instead, it is absorbed into the same geometric
    representation that defines the token's spherical position and tangent frame.
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
            self.fourier_weights = nn.Parameter(fourier_weights)
        else:
            self.fourier_weights = nn.Parameter(
                torch.randn(num_fourier_features, self.input_dim) * 0.5
            )

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
        """
        Args:
            angles_deg: [..., 2] in degrees, ordered as [lon/theta, lat/phi].
        Returns:
            [..., 3] unit vectors on S^2, using the same ERP convention as the dataset.
        """
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
        """
        Args:
            angles_deg: [B, N, 2] in degrees, ordered as [theta, phi].
            gauge_angles_deg: [B, N] or [B, N, 1] in degrees.
        Returns:
            [B, N, 9] flattened oriented frame:
                concat(x_g, y_g, p_i)
            where:
                p_i  : spherical patch center,
                x_g  : gauge-rotated tangent x/right axis,
                y_g  : gauge-rotated tangent y/up axis.
        """
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

        # Patch center direction p_i.
        p = torch.stack([
            cos_phi * cos_theta,
            -cos_phi * sin_theta,
            sin_phi,
        ], dim=-1)

        # Canonical tangent frame at p_i, consistent with dataset extraction.
        x_axis = torch.stack([
            -sin_theta,
            -cos_theta,
            torch.zeros_like(theta),
        ], dim=-1)
        y_axis = torch.stack([
            -sin_phi * cos_theta,
            sin_phi * sin_theta,
            cos_phi,
        ], dim=-1)

        # Rotate tangent frame by local gauge psi_i.
        c = torch.cos(psi).unsqueeze(-1)
        s = torch.sin(psi).unsqueeze(-1)
        x_g = c * x_axis + s * y_axis
        y_g = -s * x_axis + c * y_axis

        # Numerical safety near poles / mixed precision.
        p = torch.nn.functional.normalize(p, dim=-1)
        x_g = torch.nn.functional.normalize(x_g, dim=-1)
        y_g = torch.nn.functional.normalize(y_g, dim=-1)

        frame = torch.cat([x_g, y_g, p], dim=-1)  # [B, N, 9]
        # Frobenius norm of a perfect frame is sqrt(3); normalization keeps the
        # Fourier projection scale stable without removing orientation.
        frame = torch.nn.functional.normalize(frame, dim=-1)
        return frame

    def forward(
        self,
        angles: torch.Tensor,
        gauge_angles: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            angles: [B, N, 2] in degrees, ordered as [lon/theta, lat/phi].
            gauge_angles: optional [B, N] or [B, N, 1] in degrees.
        Returns:
            [B, N, D]
        """
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
    ALiBi-style spherical relative attention bias, optionally augmented with an
    oriented-frame similarity term.

    Geometry term:
        b_ij^h = -alpha_h * arccos(p_i^T p_j)

    Optional oriented-frame term:
        b_ij^h += beta_h * 0.5 * (x_i^T x_j + y_i^T y_j)

    The optional term uses the actual gauge-rotated tangent axes rather than an
    independent cos(psi_i - psi_j) scalar. This keeps psi_i coupled to the local
    spherical frame.
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
        # Kept as "gauge_bias" for CLI/checkpoint naming compatibility, but it is
        # now an oriented-frame relative bias.
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

        p = torch.stack([
            cos_phi * cos_theta,
            -cos_phi * sin_theta,
            sin_phi,
        ], dim=-1)
        x_axis = torch.stack([
            -sin_theta,
            -cos_theta,
            torch.zeros_like(theta),
        ], dim=-1)
        y_axis = torch.stack([
            -sin_phi * cos_theta,
            sin_phi * sin_theta,
            cos_phi,
        ], dim=-1)

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
        """
        Args:
            angles: [B, N, 2] in degrees.
            gauge_angles: optional [B, N] or [B, N, 1] in degrees.
            include_cls: prepend zero row/column for CLS.
        Returns:
            bias: [B, H, N(+1), N(+1)]
        """
        p = self.angles_to_unit_vec(angles)  # [B, N, 3]
        cos_sim = torch.einsum("bnd,bmd->bnm", p, p).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        ang_dist = torch.arccos(cos_sim)  # [B, N, N]

        slopes = torch.exp(self.log_slopes)  # [H]
        bias = -slopes.view(1, -1, 1, 1) * ang_dist.unsqueeze(1)  # [B, H, N, N]

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
                B, H, N + 1, N + 1,
                device=bias.device,
                dtype=bias.dtype,
            )
            bias_padded[:, :, 1:, 1:] = bias
            return bias_padded
        return bias


class AttentionWithBias(nn.Module):
    """Standard MHSA with an additive attention bias [B, H, N, N]."""

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
    """ViT block that accepts an external attention bias."""

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


class PanoramicMAE(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        geometric_bias: bool = True,
        adaptive_masking: bool = True,
        use_angular_bias: bool = True,
        angular_bias_init_slope: float = 1.0,
        # --- GCTT / oriented SPE options ---
        use_gctt: bool = True,
        gauge_num_frequencies: int = 16,  # Deprecated; kept for CLI compatibility.
        gauge_scale_init: float = 0.02,   # Deprecated; kept for CLI compatibility.
        use_gauge_bias: bool = True,
        gauge_bias_init: float = 0.0,
    ):
        super().__init__()
        del gauge_num_frequencies, gauge_scale_init

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_embed = ViewEmbedder(in_chans=in_chans, embed_dim=embed_dim, img_size=img_size)

        self.use_gctt = use_gctt
        self.angle_pos_embed = OrientedSphericalPositionalEncoding(
            d_model=embed_dim,
            geometric_bias=geometric_bias,
            use_oriented_frame=use_gctt,
        )

        self.use_angular_bias = use_angular_bias
        self.use_gauge_bias = use_gauge_bias and use_gctt
        if use_angular_bias:
            self.enc_ang_bias = SphericalAttentionBias(
                num_heads=num_heads,
                init_max_slope=angular_bias_init_slope,
                use_gauge_bias=self.use_gauge_bias,
                gauge_bias_init=gauge_bias_init,
            )
            self.dec_ang_bias = SphericalAttentionBias(
                num_heads=decoder_num_heads,
                init_max_slope=angular_bias_init_slope,
                use_gauge_bias=self.use_gauge_bias,
                gauge_bias_init=gauge_bias_init,
            )
            block_cls = BlockWithBias
        else:
            self.enc_ang_bias = None
            self.dec_ang_bias = None
            block_cls = Block

        self.blocks = nn.ModuleList([
            block_cls(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = OrientedSphericalPositionalEncoding(
            d_model=decoder_embed_dim,
            geometric_bias=geometric_bias,
            use_oriented_frame=use_gctt,
        )
        self.decoder_blocks = nn.ModuleList([
            block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * img_size ** 2, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.adaptive_masking = adaptive_masking
        self.initialize_weights()

    def no_weight_decay(self):
        no_decay = {
            "mask_token",
            "cls_token",
            "angle_pos_embed.fourier_weights",
            "decoder_pos_embed.fourier_weights",
        }
        if self.use_angular_bias:
            no_decay.add("enc_ang_bias.log_slopes")
            no_decay.add("dec_ang_bias.log_slopes")
            if self.use_gauge_bias:
                no_decay.add("enc_ang_bias.gauge_beta")
                no_decay.add("dec_ang_bias.gauge_beta")

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and (
                "angle_pos_embed.mlp" in name
                or "decoder_pos_embed.mlp" in name
            ):
                if m.bias is not None:
                    no_decay.add(f"{name}.bias")
        return no_decay

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        w = self.view_embed.proj.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.view_embed.proj.proj.bias is not None:
            nn.init.zeros_(self.view_embed.proj.proj.bias)
        for m in [self.decoder_embed, self.decoder_pred]:
            self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _ensure_gauge(gauge_angles, angles):
        if gauge_angles is None:
            return torch.zeros(
                angles.shape[0], angles.shape[1], 1,
                device=angles.device,
                dtype=angles.dtype,
            )
        if gauge_angles.dim() == 2:
            gauge_angles = gauge_angles.unsqueeze(-1)
        return gauge_angles.to(device=angles.device, dtype=angles.dtype)

    def _compute_view_importance(self, views_patches, angles=None):
        if views_patches.dim() == 5:
            content_imp = views_patches.var(dim=[2, 3, 4])
        else:
            content_imp = views_patches.var(dim=-1)

        content_imp = torch.sqrt(content_imp + 1e-6)

        if angles is not None:
            lat_rad = torch.deg2rad(angles[..., 1])
            sphere_weight = torch.cos(lat_rad).clamp(min=0.1)
            importance = content_imp * sphere_weight
        else:
            importance = content_imp
        return importance

    def _adaptive_masking(self, B, N, len_keep, views_patches, device, angles=None):
        importance = self._compute_view_importance(views_patches, angles=angles)

        i_min = importance.min(dim=-1, keepdim=True)[0]
        i_max = importance.max(dim=-1, keepdim=True)[0]
        i_norm = (importance - i_min) / (i_max - i_min + 1e-8)

        i20 = torch.quantile(i_norm, 0.2, dim=-1, keepdim=True)
        i80 = torch.quantile(i_norm, 0.8, dim=-1, keepdim=True)
        w = torch.clamp(i80 - i20, min=0.0, max=1.0)

        m_e = torch.full_like(w, 0.6)
        m_h = 0.6 + 0.4 * w
        m_m = (m_e + m_h) / 2.0

        ranks = importance.argsort(dim=-1).argsort(dim=-1)
        q = ranks.float() / max(N - 1, 1)

        mask_ratio = torch.where(
            q <= 0.2,
            m_e,
            torch.where(q > 0.8, m_h, m_m),
        )

        keep_prob = 1.0 - mask_ratio
        expected = keep_prob.sum(dim=-1, keepdim=True)
        scale = len_keep / (expected + 1e-8)
        keep_prob_scaled = torch.clamp(keep_prob * scale, min=1e-6, max=1.0 - 1e-6)

        log_prob = torch.log(keep_prob_scaled)
        u = torch.rand_like(log_prob).clamp(1e-8, 1.0 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        perturbed_scores = log_prob + gumbel
        ids_keep = perturbed_scores.topk(len_keep, dim=-1).indices
        return ids_keep

    def forward_encoder(self, x, visible_angles=None, visible_gauge_angles=None):
        attn_bias = None
        if self.use_angular_bias and visible_angles is not None:
            attn_bias = self.enc_ang_bias(
                visible_angles,
                gauge_angles=visible_gauge_angles if self.use_gctt else None,
                include_cls=True,
            )

        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)
        return self.norm(x)

    def forward_decoder(self, x, ids_restore, all_angles, all_gauge_angles=None):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        if self.use_gctt:
            all_gauge_angles = self._ensure_gauge(all_gauge_angles, all_angles)

        cls_pe = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        patch_pe = self.decoder_pos_embed(
            all_angles,
            gauge_angles=all_gauge_angles if self.use_gctt else None,
        )
        x = x + torch.cat([cls_pe, patch_pe], dim=1)

        attn_bias = None
        if self.use_angular_bias:
            attn_bias = self.dec_ang_bias(
                all_angles,
                gauge_angles=all_gauge_angles if self.use_gctt else None,
                include_cls=True,
            )

        for blk in self.decoder_blocks:
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)
        x = self.decoder_pred(self.decoder_norm(x))
        return x[:, 1:, :]

    def forward_loss(self, views, pred, mask):
        target = views.view(views.shape[0], views.shape[1], -1)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward(self, views, angles, gauge_angles=None, mask_ratio=0.75):
        """
        Args:
            views: [B, N, C, H, W]
            angles: [B, N, 2] in degrees.
            gauge_angles: optional [B, N] or [B, N, 1] in degrees. If omitted,
                zero gauge is used, so the model remains backward-compatible.
            mask_ratio: MAE mask ratio.

        Token input:
            z_i = PatchEmbed(P_i(theta_i, phi_i, psi_i))
                  + SPE(theta_i, phi_i, psi_i)

        There is no standalone GE(psi_i).
        """
        B, N, C, H, W = views.shape
        gauge_angles = self._ensure_gauge(gauge_angles, angles) if self.use_gctt else None

        x = self.view_embed(views.view(B * N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(
            angles,
            gauge_angles=gauge_angles if self.use_gctt else None,
        )

        len_keep = int(N * (1 - mask_ratio))
        len_keep = max(1, min(N, len_keep))

        if self.adaptive_masking and self.training:
            ids_keep = self._adaptive_masking(B, N, len_keep, views, x.device, angles=angles)
            mask_bool = torch.ones(B, N, device=views.device, dtype=torch.bool)
            mask_bool.scatter_(1, ids_keep, False)
            N_mask = N - len_keep
            ids_mask = mask_bool.nonzero()[:, 1].view(B, N_mask)
            ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            mask = mask_bool.float()
        else:
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            mask = torch.ones([B, N], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]),
        )

        visible_angles = torch.gather(
            angles,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, angles.shape[2]),
        )
        if self.use_gctt:
            visible_gauge_angles = torch.gather(
                gauge_angles,
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, gauge_angles.shape[2]),
            )
        else:
            visible_gauge_angles = None

        cls_token = self.cls_token.expand(B, -1, -1)
        x_input = torch.cat((cls_token, x_masked), dim=1)
        latent_with_cls = self.forward_encoder(
            x_input,
            visible_angles=visible_angles,
            visible_gauge_angles=visible_gauge_angles,
        )
        pred = self.forward_decoder(
            latent_with_cls,
            ids_restore,
            angles,
            all_gauge_angles=gauge_angles,
        )

        loss = self.forward_loss(views, pred, mask)
        return loss, pred, mask


# Model Factory

def vit_base_patch16(**kwargs):
    return PanoramicMAE(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large_patch16(**kwargs):
    return PanoramicMAE(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_huge_patch14(**kwargs):
    return PanoramicMAE(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
