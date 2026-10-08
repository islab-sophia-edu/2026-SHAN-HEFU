import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed


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
# 2. Pure additive 3D spherical positional encoding
# =========================================================

class AnglePositionalEncoding(nn.Module):
    """
    Pure 3DPE spherical positional encoding.

    angles[..., 0] = longitude/theta in degrees
    angles[..., 1] = latitude/phi in degrees

    lon/lat -> p=(x,y,z) on S^2:
        x = cos(lat) * cos(lon)
        y = -cos(lat) * sin(lon)
        z = sin(lat)

    PE(p) = MLP([sin(2*pi*pW^T), cos(2*pi*pW^T)])

    `geometric_bias` only controls Fourier-weight initialization.
    It does not add pairwise attention bias.
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
        lon = torch.deg2rad(angles_deg[..., 0])
        lat = torch.deg2rad(angles_deg[..., 1])
        cos_lat = torch.cos(lat)
        x = cos_lat * torch.cos(lon)
        y = -cos_lat * torch.sin(lon)
        z = torch.sin(lat)
        return torch.nn.functional.normalize(torch.stack([x, y, z], dim=-1), dim=-1)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        coords_3d = self.angles_to_unit_vec(angles)
        weights = self.fourier_weights.to(device=coords_3d.device, dtype=coords_3d.dtype)
        phase = 2.0 * math.pi * torch.matmul(coords_3d, weights.T)
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
    Build timm VisionTransformer Block across timm versions.
    Newer timm uses `proj_drop`; older MAE/timm uses `drop`.
    """
    sig = inspect.signature(Block)
    kwargs = {
        "dim": embed_dim,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": qkv_bias,
        "norm_layer": norm_layer,
    }
    if "attn_drop" in sig.parameters:
        kwargs["attn_drop"] = attn_drop_rate
    if "drop_path" in sig.parameters:
        kwargs["drop_path"] = drop_path_rate
    if "proj_drop" in sig.parameters:
        kwargs["proj_drop"] = drop_rate
    elif "drop" in sig.parameters:
        kwargs["drop"] = drop_rate
    return Block(**kwargs)


# =========================================================
# 3. Compact original-style decoder
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


class CompactNormalHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1, eps=1e-6)


# =========================================================
# 4. PanoNormal: clean 3DPE-only ablation
# =========================================================

class PanoNormal(nn.Module):
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
        cnn_base_dim=64,
        **kwargs,
    ):
        super().__init__()
        # Ignore legacy/task-specific extras if old launch commands pass them.
        del kwargs

        self.view_embed = nn.Sequential()
        self.view_embed.add_module(
            "proj",
            PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            ),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.angle_pos_embed = AnglePositionalEncoding(
            d_model=embed_dim,
            geometric_bias=geometric_bias,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            _make_standard_vit_block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=dpr[i],
                norm_layer=norm_layer,
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
        self.normal_head = CompactNormalHead(in_channels=64)

        self._last_aux_outputs = []
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

        # Clean 3DPE encoder path: view embedding + additive theta/phi 3DPE only.
        x = self.view_embed(views.view(B * N, C, H_p, W_p)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        raw_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                raw_features.append(self.norm(x)[:, 1:, :])

        f3 = self._to_grid(raw_features[0])
        f5 = self._to_grid(raw_features[1])
        f7 = self._to_grid(raw_features[2])
        f11 = self._to_grid(raw_features[3])

        x = self.neck_conv(f11)

        s1_vit = self.skip1_up(f7)
        s1_cnn = F.interpolate(cnn_low, size=s1_vit.shape[-2:], mode="bilinear", align_corners=False)
        s1_cnn = self.cnn_proj1(s1_cnn)
        x = self.up1(x, torch.cat([s1_vit, s1_cnn], dim=1))

        s2_vit = self.skip2_up(f5)
        s2_cnn = F.interpolate(cnn_mid, size=s2_vit.shape[-2:], mode="bilinear", align_corners=False)
        s2_cnn = self.cnn_proj2(s2_cnn)
        x = self.up2(x, torch.cat([s2_vit, s2_cnn], dim=1))

        s3_vit = self.skip3_up(f3)
        s3_cnn = F.interpolate(cnn_high, size=s3_vit.shape[-2:], mode="bilinear", align_corners=False)
        s3_cnn = self.cnn_proj3(s3_cnn)
        x = self.up3(x, torch.cat([s3_vit, s3_cnn], dim=1))

        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        normal = self.normal_head(x)

        patch_h = self.output_size[0] // self.grid_h
        patch_w = self.output_size[1] // self.grid_w
        x = normal.view(B, 3, self.grid_h, patch_h, self.grid_w, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, N, 3, patch_h, patch_w)
        return F.normalize(x, p=2, dim=2, eps=1e-6)

    def get_aux_outputs(self):
        return []


def vit_base_patch16(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return PanoNormal(embed_dim=768, depth=12, num_heads=12, **kwargs)


def vit_large_patch16(**kwargs):
    return PanoNormal(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def vit_huge_patch14(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return PanoNormal(embed_dim=1280, depth=32, num_heads=16, **kwargs)
