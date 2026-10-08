from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class ViewEmbedder(nn.Module):
    """Embed each tangent-plane view as one ViT token."""

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
    Standard deterministic 1D sine-cosine positional embedding.

    Args:
        embed_dim: output channel count for this axis.
        pos: flattened coordinate tensor, shape [M].

    Returns:
        Tensor with shape [M, embed_dim].
    """
    if embed_dim <= 0:
        return pos.new_zeros((pos.numel(), 0))

    half_dim = embed_dim // 2
    if half_dim == 0:
        # Degenerate case for very small dimensions.
        return torch.sin(pos).unsqueeze(-1)

    omega = torch.arange(half_dim, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000.0 ** (omega / max(half_dim, 1)))
    out = pos.float().reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    # If embed_dim is odd, pad one zero channel to keep the requested dimension.
    if emb.shape[1] < embed_dim:
        emb = torch.cat([emb, emb.new_zeros((emb.shape[0], embed_dim - emb.shape[1]))], dim=1)
    elif emb.shape[1] > embed_dim:
        emb = emb[:, :embed_dim]
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_height: int, grid_width: int) -> torch.Tensor:
    """
    Build fixed absolute 2D sine-cosine PE for the regular panoramic token grid.

    Token order must match PanoramicDataset._generate_grid_angles_rad():
    row-major [latitude row, longitude column], i.e. shape grid_height x grid_width.

    This is intentionally planar 2DPE: it uses grid row/column indices and ignores
    spherical angles, longitude periodicity, and 3D unit-vector mapping. That makes it
    a clean replacement ablation for the original 3D Fourier SPE/3DPE.
    """
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"grid_height and grid_width must be positive, got {grid_height}, {grid_width}")

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
    Fixed planar 2D sine-cosine positional encoding for 2DPE ablation.

    This module deliberately has no learnable parameters and does not use `angles`.
    The `angles` argument is accepted only to keep the same forward interface as the
    3DPE/SPE implementation.
    """

    def __init__(self, d_model: int, grid_height: int):
        super().__init__()
        self.d_model = d_model
        self.grid_height = grid_height
        self.grid_width = 2 * grid_height
        self.num_patches = self.grid_height * self.grid_width

        pos_embed = get_2d_sincos_pos_embed(d_model, self.grid_height, self.grid_width)
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
                "Pass the same --grid_height to dataset and model."
            )

        return self.pos_embed.to(device=angles.device, dtype=angles.dtype).unsqueeze(0).expand(B, -1, -1)


class PanoramicMAE(nn.Module):
    """
    PanoMAE 2DPE ablation.

    Kept identical to the 3DPE pretrain path except positional encoding:
      - tangent-plane tokenization is unchanged;
      - standard ViT/MAE blocks are unchanged;
      - adaptive masking is unchanged;
      - 3D unit-vector Fourier SPE/3DPE is replaced by fixed planar 2D sin-cos PE.
    """

    def __init__(self, img_size=224, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, grid_height: int = 4,
                 geometric_bias: bool = True, adaptive_masking: bool = True):
        super().__init__()
        # `geometric_bias` is kept only for CLI/checkpoint compatibility with the
        # 3DPE script. It has no effect in this 2DPE ablation.
        del geometric_bias

        self.grid_height = grid_height
        self.grid_width = 2 * grid_height
        self.num_patches = self.grid_height * self.grid_width

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_embed = ViewEmbedder(in_chans=in_chans, embed_dim=embed_dim, img_size=img_size)
        self.angle_pos_embed = Fixed2DSinusoidalPositionalEncoding(
            d_model=embed_dim,
            grid_height=grid_height,
        )

        # Same as the clean 3DPE ablation: standard MAE/ViT blocks, no pairwise PB.
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = Fixed2DSinusoidalPositionalEncoding(
            d_model=decoder_embed_dim,
            grid_height=grid_height,
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * img_size ** 2, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.adaptive_masking = adaptive_masking
        self.initialize_weights()

    def no_weight_decay(self):
        # 2DPE is a fixed buffer, not a parameter. Only tokens are exempted.
        return {"mask_token", "cls_token"}

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        w = self.view_embed.proj.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def _compute_view_importance(self, views_patches, angles=None):
        # Same as 3DPE version: content variance with optional spherical area weight.
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

        mask_ratio = torch.where(q <= 0.2, m_e, torch.where(q > 0.8, m_h, m_m))

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

    def forward_encoder(self, x):
        """Standard MAE encoder: CLS + visible tokens, without attention position bias."""
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_decoder(self, x, ids_restore, all_angles):
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

        cls_pe = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        patch_pe = self.decoder_pos_embed(all_angles).to(dtype=x.dtype)
        x = x + torch.cat([cls_pe, patch_pe], dim=1)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_pred(self.decoder_norm(x))
        return x[:, 1:, :]

    def forward_loss(self, views, pred, mask):
        """
        views: [B, N, C, H, W]
        pred:  [B, N, P*P*C]
        mask:  [B, N]  (1 for masked, 0 for visible)
        """
        target = views.view(views.shape[0], views.shape[1], -1)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward(self, views, angles, mask_ratio=0.75):
        B, N, C, H, W = views.shape
        if N != self.num_patches:
            raise ValueError(
                f"Input token count N={N} does not match model 2DPE grid "
                f"{self.grid_height}x{self.grid_width}={self.num_patches}."
            )

        x = self.view_embed(views.view(B * N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles).to(dtype=x.dtype)
        len_keep = int(N * (1 - mask_ratio))
        len_keep = max(1, min(len_keep, N - 1))

        if self.adaptive_masking and self.training:
            ids_keep = self._adaptive_masking(B, N, len_keep, views, x.device, angles=angles)
            mask_bool = torch.ones(B, N, device=views.device, dtype=torch.bool)
            mask_bool.scatter_(1, ids_keep, False)
            n_mask = N - len_keep
            ids_mask = mask_bool.nonzero()[:, 1].view(B, n_mask)
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

        cls_token = self.cls_token.expand(B, -1, -1)
        x_input = torch.cat((cls_token, x_masked), dim=1)
        latent_with_cls = self.forward_encoder(x_input)
        pred = self.forward_decoder(latent_with_cls, ids_restore, angles)

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
