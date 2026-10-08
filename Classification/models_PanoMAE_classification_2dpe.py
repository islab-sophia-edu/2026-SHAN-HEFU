import inspect
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


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Deterministic 1D sine-cosine positional embedding.

    This is the standard absolute sinusoidal PE form used as the planar 2DPE
    ablation baseline. It deliberately uses grid coordinates, not spherical angles.
    """
    if embed_dim <= 0:
        return pos.new_zeros((pos.numel(), 0))

    half_dim = embed_dim // 2
    if half_dim == 0:
        return torch.sin(pos.float()).unsqueeze(-1)

    omega = torch.arange(half_dim, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000.0 ** (omega / max(half_dim, 1)))
    out = pos.float().reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    if emb.shape[1] < embed_dim:
        emb = torch.cat(
            [emb, emb.new_zeros((emb.shape[0], embed_dim - emb.shape[1]))],
            dim=1,
        )
    elif emb.shape[1] > embed_dim:
        emb = emb[:, :embed_dim]
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_height: int, grid_width: int) -> torch.Tensor:
    """
    Build fixed absolute 2D sine-cosine PE for the tangent-view token grid.

    Token order matches PanoClassificationDataset._generate_grid_angles_rad():
    row-major order over [latitude row, longitude column].

    This is intentionally planar 2DPE:
      - no lon/lat -> 3D unit-vector mapping;
      - no Fourier weights in R^{Kx3};
      - no MLP positional encoder;
      - no pairwise PB / angular attention bias.
    """
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(
            f"grid_height and grid_width must be positive, got {grid_height}, {grid_width}"
        )

    y = torch.arange(grid_height, dtype=torch.float32)
    x = torch.arange(grid_width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    dim_y = embed_dim // 2
    dim_x = embed_dim - dim_y
    emb_y = _get_1d_sincos_pos_embed(dim_y, yy)
    emb_x = _get_1d_sincos_pos_embed(dim_x, xx)
    return torch.cat([emb_y, emb_x], dim=1)


class Fixed2DSinusoidalPositionalEncoding(nn.Module):
    """
    Fixed planar 2D sine-cosine positional encoding for classification 2DPE ablation.

    The forward method accepts `angles` only to preserve the 3DPE model interface:
        model(views, angles)
    The returned PE ignores the actual angle values and depends only on token index
    in the regular grid.
    """

    def __init__(self, d_model: int, grid_height: int):
        super().__init__()
        self.d_model = int(d_model)
        self.grid_height = int(grid_height)
        self.grid_width = 2 * self.grid_height
        self.num_patches = self.grid_height * self.grid_width

        pos_embed = get_2d_sincos_pos_embed(
            self.d_model,
            self.grid_height,
            self.grid_width,
        )
        self.register_buffer("pos_embed", pos_embed, persistent=True)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles is None:
            return self.pos_embed.unsqueeze(0)

        if angles.dim() < 2:
            raise ValueError(f"angles should have shape [B, N, 2], got {tuple(angles.shape)}")

        B, N = angles.shape[0], angles.shape[1]
        if N != self.num_patches:
            raise ValueError(
                f"2DPE token count mismatch: got N={N}, but grid_height={self.grid_height} "
                f"implies {self.num_patches} tokens ({self.grid_height}x{self.grid_width}). "
                "Use the same --grid_height for dataset and model."
            )

        return self.pos_embed.to(device=angles.device, dtype=angles.dtype).unsqueeze(0).expand(B, -1, -1)


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
    PanoMAE downstream classifier: clean 2DPE ablation.

    It keeps the classification architecture and data interface identical to the 3DPE
    classifier, but replaces additive 3D spherical positional encoding with fixed
    planar 2D sine-cosine positional encoding.

    Input:
        views:  [B, N, C, H, W]
        angles: [B, N, 2], degrees, [lon, lat]. Accepted for interface compatibility;
                ignored by the 2DPE module.

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
        grid_height: int = 4,
        geometric_bias=True,
    ):
        super().__init__()
        # Kept only for CLI/checkpoint compatibility with 3DPE scripts.
        del geometric_bias

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grid_height = int(grid_height)
        self.grid_width = 2 * self.grid_height
        self.num_patches = self.grid_height * self.grid_width

        self.view_embed = ViewEmbedder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            img_size=img_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.angle_pos_embed = Fixed2DSinusoidalPositionalEncoding(
            d_model=embed_dim,
            grid_height=self.grid_height,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Clean 2DPE ablation: standard ViT blocks, no PB / angular attention bias.
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
        # 2DPE is a fixed buffer, not a learnable parameter.
        return {"cls_token"}

    def forward_features(self, views, angles):
        B, N, C, H, W = views.shape
        if N != self.num_patches:
            raise ValueError(
                f"Input token count N={N} does not match model 2DPE grid "
                f"{self.grid_height}x{self.grid_width}={self.num_patches}."
            )

        x = self.view_embed(views.view(B * N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles).to(dtype=x.dtype)

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
