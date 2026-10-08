import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed, Mlp, DropPath


# =========================================================
# 1. Small CNN detail branch, kept close to your original decoder
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
# 2. PB spherical PE components, key-compatible with original PB pretrain
# =========================================================

class AnglePositionalEncoding(nn.Module):
    """
    Original PB positional encoding: lon/lat -> S^2 unit vector -> Random
    Fourier Features -> MLP. There is no gauge angle, no oriented local frame,
    and no standalone GE(psi).
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
            self.fourier_weights = nn.Parameter(torch.randn(num_fourier_features, 3) * 0.5)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_fourier_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _build_stratified_fourier_weights(num_features: int, max_frequency: float) -> torch.Tensor:
        dirs = torch.randn(num_features, 3)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)
        freqs = torch.linspace(0.5, max_frequency, num_features).unsqueeze(-1)
        return dirs * freqs

    @staticmethod
    def angles_to_unit_vec(angles_deg: torch.Tensor) -> torch.Tensor:
        lon_rad = torch.deg2rad(angles_deg[..., 0])
        lat_rad = torch.deg2rad(angles_deg[..., 1])
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = -torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        coords_3d = torch.stack([x, y, z], dim=-1)
        return torch.nn.functional.normalize(coords_3d, dim=-1)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        coords_3d = self.angles_to_unit_vec(angles)
        weights = self.fourier_weights.to(device=coords_3d.device, dtype=coords_3d.dtype)
        phase = 2.0 * math.pi * torch.matmul(coords_3d, weights.T)
        fourier_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.mlp(fourier_features)


class SphericalAttentionBias(nn.Module):
    """
    Original PB ALiBi-style angular distance bias on S^2:
        bias_h(i,j) = -slope_h * arccos(p_i · p_j)
    No gauge/frame similarity term is used.
    """
    def __init__(self, num_heads: int, init_max_slope: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([init_max_slope / (2 ** i) for i in range(num_heads)], dtype=torch.float32)
        self.log_slopes = nn.Parameter(torch.log(slopes))

    @staticmethod
    def angles_to_unit_vec(angles_deg: torch.Tensor) -> torch.Tensor:
        return AnglePositionalEncoding.angles_to_unit_vec(angles_deg)

    def forward(self, angles: torch.Tensor, include_cls: bool = True) -> torch.Tensor:
        p = self.angles_to_unit_vec(angles)
        cos_sim = torch.einsum("bnd,bmd->bnm", p, p).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        ang_dist = torch.arccos(cos_sim)
        slopes = torch.exp(self.log_slopes).to(device=angles.device, dtype=ang_dist.dtype)
        bias = -slopes.view(1, -1, 1, 1) * ang_dist.unsqueeze(1)
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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
        self.attn = AttentionWithBias(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, attn_bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias=attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =========================================================
# 3. Deeper depth-specific encoder, memory-light
# =========================================================

class GridTokenMixerBlock(nn.Module):
    """
    Depth-specific encoder block over the N=tokens grid.

    This is intentionally not another full self-attention stack. For grid_height=16,
    N=512 and huge dim=1280, extra global attention is expensive and likely to OOM.
    This block is ConvNeXt-style: LayerNorm + depthwise spatial grid convolution
    + channel MLP + residual gating. It is much deeper than a thin adapter but
    keeps the pretrained PB encoder intact and key-compatible.
    """

    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        mlp_ratio: float = 0.5,
        kernel_size: int = 3,
        drop_path: float = 0.0,
        init_scale: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        hidden = max(int(dim * mlp_ratio), 64)
        pad = kernel_size // 2

        self.norm1 = nn.LayerNorm(dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_spatial = nn.Parameter(torch.ones(dim) * float(init_scale))
        self.gamma_mlp = nn.Parameter(torch.ones(dim) * float(init_scale))

    def forward(self, tokens):
        # tokens: [B, N, D], no CLS.
        B, N, D = tokens.shape
        if N != self.grid_h * self.grid_w:
            raise ValueError(f"Expected {self.grid_h*self.grid_w} tokens, got {N}.")

        y = self.norm1(tokens)
        y = y.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)
        y = self.spatial(y)
        y = y.flatten(2).transpose(1, 2)
        tokens = tokens + self.drop_path(self.gamma_spatial.view(1, 1, -1) * y)

        z = self.mlp(self.norm2(tokens))
        tokens = tokens + self.drop_path(self.gamma_mlp.view(1, 1, -1) * z)
        return tokens


class DepthTaskEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        depth: int = 6,
        mlp_ratio: float = 0.5,
        kernel_size: int = 3,
        drop_path_rate: float = 0.0,
        init_scale: float = 1e-2,
    ):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList([
            GridTokenMixerBlock(
                dim=dim,
                grid_h=grid_h,
                grid_w=grid_w,
                mlp_ratio=mlp_ratio,
                kernel_size=kernel_size,
                drop_path=float(dpr[i]),
                init_scale=init_scale,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


# =========================================================
# 4. Compact original-style decoder
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
        return self.pixel_shuffle(self.conv1x1(x))


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
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 1), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, vit_feat, cnn_feat):
        combined = self.conv(torch.cat([vit_feat, cnn_feat], dim=1))
        return combined * self.attention(combined)


class CompactDepthHead(nn.Module):
    """
    Small depth head. The decoder remains close to the original model; only the
    output parameterization is selectable.

    - sigmoid: old linear normalized depth.
    - log: log-spaced continuous depth, better calibrated for max_depth=100.
    """

    def __init__(self, in_channels=64, max_depth=100.0, min_depth=0.05, mode="log"):
        super().__init__()
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)
        self.mode = str(mode).lower()
        if self.mode == "bins":
            # v5 deliberately returns to the compact original-style decoder.
            # Keep bins as a CLI-compatible alias so old commands still run.
            self.mode = "log"
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.register_buffer("bin_edges_m", torch.empty(0), persistent=False)

    def forward(self, x):
        raw = self.net(x)
        if self.mode == "sigmoid":
            return torch.sigmoid(raw).clamp(0.0, 1.0), None
        if self.mode == "log":
            t = torch.sigmoid(raw)
            log_min = math.log(self.min_depth)
            log_max = math.log(self.max_depth)
            depth_m = torch.exp(log_min + t * (log_max - log_min))
            return (depth_m / self.max_depth).clamp(0.0, 1.0), None
        raise ValueError(f"Unsupported depth_head_type for compact decoder: {self.mode}. Use log or sigmoid.")


# =========================================================
# 5. PB PanoDepth with retained pretrained PB encoder + deeper task encoder
# =========================================================

class PanoDepth(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=32,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        geometric_bias=True,
        grid_height=16,
        output_size=(832, 1664),
        drop_path_rate=0.0,
        use_angular_bias=True,
        angular_bias_init_slope=1.0,
        max_depth=100.0,
        min_depth=0.05,
        depth_head_type="log",
        depth_encoder_depth=8,
        stage_encoder_depth=1,
        depth_encoder_mlp_ratio=0.5,
        depth_encoder_kernel_size=3,
        depth_encoder_drop_path=0.0,
        depth_encoder_init_scale=1e-2,
        cnn_base_dim=64,
        **kwargs,
    ):
        super().__init__()
        # Kept for CLI/checkpoint compatibility; intentionally unused in PB ablation.
        kwargs.pop("num_depth_bins", None)
        kwargs.pop("use_pretrained_mae_decoder", None)
        kwargs.pop("decoder_embed_dim", None)
        kwargs.pop("decoder_depth", None)
        kwargs.pop("decoder_num_heads", None)
        kwargs.pop("use_pb", None)
        kwargs.pop("use_gauge_bias", None)
        kwargs.pop("gauge_bias_init", None)
        del kwargs

        self.view_embed = nn.Sequential()
        self.view_embed.add_module("proj", PatchEmbed(img_size, patch_size, in_chans, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # PB-only: no gauge_angles, no oriented spherical frame, no gauge bias.
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)

        self.use_angular_bias = use_angular_bias
        if use_angular_bias:
            self.enc_ang_bias = SphericalAttentionBias(
                num_heads=num_heads,
                init_max_slope=angular_bias_init_slope,
            )
            block_cls = BlockWithBias
        else:
            self.enc_ang_bias = None
            block_cls = Block

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_cls(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
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
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)
        self.depth_head_type = str(depth_head_type).lower()
        self.last_depth_bin_logits = None

        # Kept identical to the PB depth variant for a fair PB-vs-PB ablation:
        # same task encoder/decoder capacity, different geometric encoding only.
        self.depth_encoder = DepthTaskEncoder(
            dim=embed_dim,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            depth=int(depth_encoder_depth),
            mlp_ratio=float(depth_encoder_mlp_ratio),
            kernel_size=int(depth_encoder_kernel_size),
            drop_path_rate=float(depth_encoder_drop_path),
            init_scale=float(depth_encoder_init_scale),
        )
        self.stage_adapters = nn.ModuleList([
            DepthTaskEncoder(
                dim=embed_dim,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                depth=int(stage_encoder_depth),
                mlp_ratio=float(depth_encoder_mlp_ratio),
                kernel_size=int(depth_encoder_kernel_size),
                drop_path_rate=float(depth_encoder_drop_path),
                init_scale=float(depth_encoder_init_scale),
            )
            for _ in range(4)
        ])

        # Dense decoder follows the GCTT dense-cascade fusion style:
        # project CNN features to the same channel count as the ViT skip,
        # concatenate [ViT skip, CNN skip], then learn fusion in Up/DoubleConv.
        # PB-only geometry is preserved: no gauge_angles, no oriented SPE, no gauge bias.
        self.global_detail = GlobalDetailCapture(in_chans=in_chans, base_dim=int(cnn_base_dim))
        self.cnn_proj1 = nn.Conv2d(int(cnn_base_dim) * 4, 256, kernel_size=1)
        self.cnn_proj2 = nn.Conv2d(int(cnn_base_dim) * 2, 128, kernel_size=1)
        self.cnn_proj3 = nn.Conv2d(int(cnn_base_dim), 64, kernel_size=1)

        self.neck_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)
        self.skip1_up = ReshapeUp(embed_dim, 256, scale_factor=2)
        self.skip2_up = ReshapeUp(embed_dim, 128, scale_factor=4)
        self.skip3_up = ReshapeUp(embed_dim, 64, scale_factor=8)

        self.up1 = Up(in_channels=1024, out_channels=256, bilinear=True)
        self.up2 = Up(in_channels=512, out_channels=128, bilinear=True)
        self.up3 = Up(in_channels=256, out_channels=64, bilinear=True)
        self.depth_head = CompactDepthHead(
            in_channels=64,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            mode=self.depth_head_type,
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
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
        no_decay = {"cls_token", "angle_pos_embed.fourier_weights"}
        if self.use_angular_bias:
            no_decay.add("enc_ang_bias.log_slopes")
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "angle_pos_embed.mlp" in name and m.bias is not None:
                no_decay.add(f"{name}.bias")
        return no_decay

    def _to_grid(self, tokens):
        return tokens.permute(0, 2, 1).view(tokens.shape[0], -1, self.grid_h, self.grid_w)

    def forward(self, views, angles, mask_ratio=None):
        del mask_ratio
        B, N, C, H_p, W_p = views.shape

        # Compact CNN detail path.
        x_grid = views.view(B, self.grid_h, self.grid_w, C, H_p, W_p)
        x_full = x_grid.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_full = x_full.view(B, C, self.grid_h * H_p, self.grid_w * W_p)
        cnn_high, cnn_mid, cnn_low = self.global_detail(x_full)

        # PB encoder path: image token + AnglePositionalEncoding(lon, lat).
        x = self.view_embed(views.view(B * N, C, H_p, W_p)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        attn_bias = self.enc_ang_bias(angles, include_cls=True) if self.use_angular_bias else None

        raw_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)
            if i in self.out_indices:
                raw_features.append(self.norm(x)[:, 1:, :])

        final_tokens = self.depth_encoder(raw_features[-1])
        stage_tokens = []
        for feat, adapter in zip(raw_features, self.stage_adapters):
            stage_tokens.append(adapter(feat + final_tokens))

        f3 = self._to_grid(stage_tokens[0])
        f5 = self._to_grid(stage_tokens[1])
        f7 = self._to_grid(stage_tokens[2])
        f11 = self._to_grid(final_tokens)

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
        depth, _ = self.depth_head(x)
        self.last_depth_bin_logits = None

        patch_h = self.output_size[0] // self.grid_h
        patch_w = self.output_size[1] // self.grid_w
        x = depth.view(B, 1, self.grid_h, patch_h, self.grid_w, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, N, 1, patch_h, patch_w)
        return x


def vit_base_patch16(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return PanoDepth(embed_dim=768, depth=12, num_heads=12, **kwargs)


def vit_large_patch16(**kwargs):
    return PanoDepth(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def vit_huge_patch14(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return PanoDepth(embed_dim=1280, depth=32, num_heads=16, **kwargs)
