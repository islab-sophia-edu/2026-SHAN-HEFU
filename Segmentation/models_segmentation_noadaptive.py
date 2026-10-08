import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed, Mlp, DropPath


# =========================================================
# 1. Global CNN detail extractor with panorama circular padding
# =========================================================

class CircularPad(nn.Module):
    def __init__(self, padding=1):
        super().__init__()
        self.pad = padding

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="constant", value=0)
        return x


class GlobalDetailCapture(nn.Module):
    def __init__(self, in_chans=3, base_dim=64):
        super().__init__()

        def make_layer(in_c, out_c):
            return nn.Sequential(
                CircularPad(1),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer1 = make_layer(in_chans, base_dim)
        self.layer2 = make_layer(base_dim, base_dim * 2)
        self.layer3 = make_layer(base_dim * 2, base_dim * 4)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


# =========================================================
# 2. GCTT / Oriented SPE components
# =========================================================

class OrientedSphericalPositionalEncoding(nn.Module):
    """
    GCTT positional encoding.

    If use_oriented_frame=False:
        encode only p(theta, phi) in R^3.

    If use_oriented_frame=True:
        encode the oriented local spherical frame [x_g, y_g, p] in R^9,
        where x_g/y_g are the gauge-rotated tangent axes.
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

        # Keep the same module structure as your GCTT PanoMAE checkpoint:
        # angle_pos_embed.mlp.0 / .1 / .2
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_fourier_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _build_stratified_fourier_weights(num_features: int, input_dim: int, max_frequency: float):
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

        frame = torch.cat([x_g, y_g, p], dim=-1)
        frame = torch.nn.functional.normalize(frame, dim=-1)
        return frame

    def forward(self, angles: torch.Tensor, gauge_angles: torch.Tensor = None) -> torch.Tensor:
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
    ALiBi-style spherical angular bias plus optional GCTT oriented-frame relative bias.

    Geometry:
        b_ij^h = -alpha_h * arccos(p_i^T p_j)

    Optional oriented-frame term:
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
    def angles_to_unit_vec(angles_deg):
        return OrientedSphericalPositionalEncoding.angles_to_unit_vec(angles_deg)

    @staticmethod
    def angles_to_axes(angles_deg, gauge_angles_deg=None):
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

    def forward(self, angles, gauge_angles=None, include_cls=True):
        p = self.angles_to_unit_vec(angles)
        cos_sim = torch.einsum("bnd,bmd->bnm", p, p).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        ang_dist = torch.arccos(cos_sim)

        slopes = torch.exp(self.log_slopes).to(device=angles.device, dtype=ang_dist.dtype)
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
            bias_padded = torch.zeros(B, H, N + 1, N + 1, device=bias.device, dtype=bias.dtype)
            bias_padded[:, :, 1:, 1:] = bias
            return bias_padded

        return bias


class AttentionWithBias(nn.Module):
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


# =========================================================
# 3. Segmentation decoder components
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ReshapeUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.pixel_shuffle(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =========================================================
# 4. GCTT PanoSegmenter
# =========================================================

class PanoSegmenter(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=32,
        in_chans=3,
        num_classes=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        geometric_bias=True,
        grid_height=16,
        output_size=(832, 1664),
        drop_path_rate=0.0,
        use_gctt=True,
        use_angular_bias=True,
        angular_bias_init_slope=1.0,
        use_gauge_bias=True,
        gauge_bias_init=0.0,
    ):
        super().__init__()

        # Keep key compatibility with your PanoMAE GCTT checkpoint:
        # view_embed.proj.proj.{weight,bias}
        self.view_embed = nn.Sequential()
        self.view_embed.add_module("proj", PatchEmbed(img_size, patch_size, in_chans, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.use_gctt = use_gctt
        self.angle_pos_embed = OrientedSphericalPositionalEncoding(
            d_model=embed_dim,
            geometric_bias=geometric_bias,
            use_oriented_frame=use_gctt,
        )

        self.use_angular_bias = use_angular_bias
        self.use_gauge_bias = bool(use_gauge_bias and use_gctt)

        if use_angular_bias:
            self.enc_ang_bias = SphericalAttentionBias(
                num_heads=num_heads,
                init_max_slope=angular_bias_init_slope,
                use_gauge_bias=self.use_gauge_bias,
                gauge_bias_init=gauge_bias_init,
            )
            block_cls = BlockWithBias
        else:
            self.enc_ang_bias = None
            block_cls = Block

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_cls(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if depth == 12:
            self.out_indices = [3, 5, 7, 11]
        elif depth == 24:
            self.out_indices = [5, 11, 17, 23]
        else:
            self.out_indices = [depth // 4 - 1, depth // 2 - 1, depth * 3 // 4 - 1, depth - 1]

        self.grid_h = grid_height
        self.grid_w = 2 * grid_height
        self.output_size = output_size
        self.num_classes = num_classes

        self.global_detail = GlobalDetailCapture(in_chans=in_chans, base_dim=64)

        self.cnn_proj1 = nn.Conv2d(256, 256, 1)
        self.cnn_proj2 = nn.Conv2d(128, 128, 1)
        self.cnn_proj3 = nn.Conv2d(64, 64, 1)

        self.neck_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)

        self.skip1_up = ReshapeUp(embed_dim, 256, scale_factor=2)
        self.skip2_up = ReshapeUp(embed_dim, 128, scale_factor=4)
        self.skip3_up = ReshapeUp(embed_dim, 64, scale_factor=8)

        self.up1 = Up(in_channels=1024, out_channels=256, bilinear=True)
        self.up2 = Up(in_channels=512, out_channels=128, bilinear=True)
        self.up3 = Up(in_channels=256, out_channels=64, bilinear=True)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
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

        # Match PanoMAE ViewEmbedder initialization.
        w = self.view_embed.proj.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.view_embed.proj.proj.bias is not None:
            nn.init.zeros_(self.view_embed.proj.proj.bias)

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
            if self.use_gauge_bias:
                no_decay.add("enc_ang_bias.gauge_beta")

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "angle_pos_embed.mlp" in name and m.bias is not None:
                no_decay.add(f"{name}.bias")

        return no_decay

    def forward(self, views, angles, gauge_angles=None, mask_ratio=None):
        # Noadaptive segmentation fine-tuning uses all tokens; mask_ratio is ignored.
        del mask_ratio
        B, N, C, H_p, W_p = views.shape

        if self.use_gctt:
            gauge_angles = self._ensure_gauge(gauge_angles, angles)
        else:
            gauge_angles = None

        # 1. Global CNN detail path.
        # Note: if gauge jitter is non-zero, this grid is a grid of gauge-rotated
        # tangent patches, not a physically reprojected ERP. The segmentation
        # target is still aligned because masks use the same gauge.
        x_grid = views.view(B, self.grid_h, self.grid_w, C, H_p, W_p)
        x_full = x_grid.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_full = x_full.view(B, C, self.grid_h * H_p, self.grid_w * W_p)

        cnn_feats = self.global_detail(x_full)
        cnn_high = cnn_feats[0]
        cnn_mid = cnn_feats[1]
        cnn_low = cnn_feats[2]

        # 2. GCTT ViT encoder.
        x = views.view(B * N, C, H_p, W_p)
        x = self.view_embed(x).view(B, N, -1)

        x = x + self.angle_pos_embed(
            angles,
            gauge_angles=gauge_angles if self.use_gctt else None,
        )

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        attn_bias = None
        if self.use_angular_bias:
            attn_bias = self.enc_ang_bias(
                angles,
                gauge_angles=gauge_angles if self.use_gctt else None,
                include_cls=True,
            )

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)
            if i in self.out_indices:
                features.append(self.norm(x))

        def to_grid(t):
            t = t[:, 1:, :]
            return t.permute(0, 2, 1).view(B, -1, self.grid_h, self.grid_w)

        f3 = to_grid(features[0])
        f5 = to_grid(features[1])
        f7 = to_grid(features[2])
        f11 = to_grid(features[3])

        # 3. Decoder fusion.
        x = self.neck_conv(f11)

        s1_vit = self.skip1_up(f7)
        s1_cnn = F.interpolate(cnn_low, size=s1_vit.shape[-2:], mode="bilinear", align_corners=False)
        s1_cnn = self.cnn_proj1(s1_cnn)
        s1_combined = torch.cat([s1_vit, s1_cnn], dim=1)
        x = self.up1(x, s1_combined)

        s2_vit = self.skip2_up(f5)
        s2_cnn = F.interpolate(cnn_mid, size=s2_vit.shape[-2:], mode="bilinear", align_corners=False)
        s2_cnn = self.cnn_proj2(s2_cnn)
        s2_combined = torch.cat([s2_vit, s2_cnn], dim=1)
        x = self.up2(x, s2_combined)

        s3_vit = self.skip3_up(f3)
        s3_cnn = F.interpolate(cnn_high, size=s3_vit.shape[-2:], mode="bilinear", align_corners=False)
        s3_cnn = self.cnn_proj3(s3_cnn)
        s3_combined = torch.cat([s3_vit, s3_cnn], dim=1)
        x = self.up3(x, s3_combined)

        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        logits = self.head(x)

        patch_h = self.output_size[0] // self.grid_h
        patch_w = self.output_size[1] // self.grid_w
        x = logits.view(B, self.num_classes, self.grid_h, patch_h, self.grid_w, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, N, self.num_classes, patch_h, patch_w)
        return x


def vit_huge_patch14(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return PanoSegmenter(embed_dim=1280, depth=32, num_heads=16, **kwargs)


def vit_large_patch16(**kwargs):
    return PanoSegmenter(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def vit_base_patch16(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return PanoSegmenter(embed_dim=768, depth=12, num_heads=12, **kwargs)
