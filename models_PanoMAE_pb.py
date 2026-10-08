from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import math
from timm.models.vision_transformer import Mlp, DropPath

class ViewEmbedder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, img_size=224):
        super().__init__()
        from timm.models.layers import PatchEmbed
        self.proj = PatchEmbed(
            img_size=img_size, patch_size=img_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )

    def forward(self, x):
        return self.proj(x)
    
#Random Fourier Features
class AnglePositionalEncoding(nn.Module):
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
    ALiBi-style angular distance bias for spherical attention.
    
    For each pair of tokens (i, j) with viewpoint directions p_i, p_j on S^2,
    computes attention bias:
        bias(i, j) = -slope_h * arccos(p_i . p_j)
    where slope_h is a learnable per-head scalar (initialized geometrically).
    
    This bias is rotation-INVARIANT: angular_distance(R p_i, R p_j) = 
    angular_distance(p_i, p_j) for any R in SO(3), complementing SPE's
    approximate rotation-equivariance with an exact rotation-invariant
    relative positional prior.
    
    Shape: bias is [B, num_heads, N+1, N+1] (with +1 for CLS token).
    """
    def __init__(self, num_heads: int, init_max_slope: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        
        # ALiBi-style geometric initialization: slopes are spaced geometrically
        # between init_max_slope and init_max_slope/2^(num_heads-1).
        # Some heads will have small slope (long-range attention),
        # others large slope (local attention).
        slopes = torch.tensor(
            [init_max_slope / (2 ** i) for i in range(num_heads)],
            dtype=torch.float32
        )
        # Use log-parameterization to keep slopes positive during training
        self.log_slopes = nn.Parameter(torch.log(slopes))
    
    @staticmethod
    def angles_to_unit_vec(angles_deg: torch.Tensor) -> torch.Tensor:
        """angles: [..., 2] (lon, lat) in degrees -> [..., 3] unit vectors on S^2."""
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
            angles: [B, N, 2] in degrees (lon, lat)
            include_cls: if True, prepend a row/col of zeros for the CLS token
        Returns:
            bias: [B, num_heads, N(+1), N(+1)]
        """
        # Compute unit vectors on S^2
        p = self.angles_to_unit_vec(angles)  # [B, N, 3]
        
        # Pairwise dot products -> cosine similarity
        # clamp for numerical stability of arccos
        cos_sim = torch.einsum('bnd,bmd->bnm', p, p).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # Angular distance in radians, in [0, pi]
        ang_dist = torch.arccos(cos_sim)  # [B, N, N]
        
        # Apply per-head slopes: bias = -slope * distance
        slopes = torch.exp(self.log_slopes)  # [num_heads], always positive
        bias = -slopes.view(1, -1, 1, 1) * ang_dist.unsqueeze(1)  # [B, H, N, N]
        
        if include_cls:
            B, H, N, _ = bias.shape
            # Pad zeros for CLS token (no bias to/from CLS)
            bias_padded = torch.zeros(B, H, N + 1, N + 1, 
                                       device=bias.device, dtype=bias.dtype)
            bias_padded[:, :, 1:, 1:] = bias
            return bias_padded
        return bias

class AttentionWithBias(nn.Module):
    """Standard MHSA with an additive attention bias [B, H, N, N]."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
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
        q, k, v = qkv.unbind(0)  # each [B, H, N, head_dim]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        if attn_bias is not None:
            # Broadcast-add bias. attn_bias: [B, H, N, N] or [1, H, N, N]
            attn = attn + attn_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockWithBias(nn.Module):
    """ViT block that accepts an external attention bias."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., 
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionWithBias(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       act_layer=nn.GELU, drop=drop)
    
    def forward(self, x, attn_bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias=attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PanoramicMAE(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, geometric_bias: bool = True, 
                 adaptive_masking: bool = True,
                 use_angular_bias: bool = True,           # ← NEW
                 angular_bias_init_slope: float = 1.0):   # ← NEW
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_embed = ViewEmbedder(in_chans=in_chans, embed_dim=embed_dim, img_size=img_size)
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)
        
        # --- NEW: Angular distance bias modules ---
        self.use_angular_bias = use_angular_bias
        if use_angular_bias:
            self.enc_ang_bias = SphericalAttentionBias(num_heads=num_heads, 
                                                       init_max_slope=angular_bias_init_slope)
            self.dec_ang_bias = SphericalAttentionBias(num_heads=decoder_num_heads,
                                                       init_max_slope=angular_bias_init_slope)
            # Use custom blocks that accept attention bias
            block_cls = BlockWithBias
        else:
            self.enc_ang_bias = None
            self.dec_ang_bias = None
            block_cls = Block  # original timm Block
        # ------------------------------------------
        
        self.blocks = nn.ModuleList([
            block_cls(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = AnglePositionalEncoding(d_model=decoder_embed_dim, geometric_bias=geometric_bias)
        self.decoder_blocks = nn.ModuleList([
            block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * img_size**2, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.adaptive_masking = adaptive_masking
        self.initialize_weights()
    
    def no_weight_decay(self):
        no_decay = {'mask_token', 'cls_token',
                    'angle_pos_embed.fourier_weights',
                    'decoder_pos_embed.fourier_weights'}
        # NEW: don't decay angular bias slopes
        if self.use_angular_bias:
            no_decay.add('enc_ang_bias.log_slopes')
            no_decay.add('dec_ang_bias.log_slopes')
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and ('angle_pos_embed.mlp' in name or 'decoder_pos_embed.mlp' in name):
                if m.bias is not None:
                    no_decay.add(f'{name}.bias')
        return no_decay
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        # Initialize view_embed (was previously skipped!)
        w = self.view_embed.proj.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        for m in [self.decoder_embed, self.decoder_pred]:
            self._init_weights(m)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _compute_view_importance(self, views_patches, angles=None):
        # Feature Variance as High-Frequency Info Proxy
        if views_patches.dim() == 5:
            content_imp = views_patches.var(dim=[2, 3, 4])
        else:
            content_imp = views_patches.var(dim=-1)
            
        content_imp = torch.sqrt(content_imp + 1e-6)
        
        # Spherical Distortion Correction: I_sphere = I * cos(lat)
        if angles is not None:
            lat_rad = torch.deg2rad(angles[..., 1])
            sphere_weight = torch.cos(lat_rad).clamp(min=0.1)
            importance = content_imp * sphere_weight
        else:
            importance = content_imp
            
        return importance

    def _adaptive_masking(self, B, N, len_keep, views_patches, device, angles=None):
        importance = self._compute_view_importance(views_patches, angles=angles)
        
        # 1. Importance Normalization [0, 1]
        i_min = importance.min(dim=-1, keepdim=True)[0]
        i_max = importance.max(dim=-1, keepdim=True)[0]
        i_norm = (importance - i_min) / (i_max - i_min + 1e-8)
        
        # 2. Distribution Spread Calculation (w = I80 - I20)
        i20 = torch.quantile(i_norm, 0.2, dim=-1, keepdim=True)
        i80 = torch.quantile(i_norm, 0.8, dim=-1, keepdim=True)
        w = torch.clamp(i80 - i20, min=0.0, max=1.0)
        
        # 3. Dynamic Mask Ratios Definition (Base m_e = 0.6)
        m_e = torch.full_like(w, 0.6)
        m_h = 0.6 + 0.4 * w
        m_m = (m_e + m_h) / 2.0
        
        # 4. Percentile-based Patch Classification
        ranks = importance.argsort(dim=-1).argsort(dim=-1)
        q = ranks.float() / (N - 1)  # Quantiles shape: (B, N)
        
        # Map Mask Ratios to Patches
        mask_ratio = torch.where(
            q <= 0.2, 
            m_e,
            torch.where(q > 0.8, m_h, m_m)
        )
        
        # 5. Keep Probability Conversion & Scaling
        keep_prob = 1.0 - mask_ratio
        expected = keep_prob.sum(dim=-1, keepdim=True)
        scale = len_keep / (expected + 1e-8)
        keep_prob_scaled = torch.clamp(keep_prob * scale, min=1e-6, max=1.0 - 1e-6)
        
        # 6. Gumbel-Max Sampling (Differentiable / Stochastic)
        log_prob = torch.log(keep_prob_scaled)
        u = torch.rand_like(log_prob).clamp(1e-8, 1.0 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        perturbed_scores = log_prob + gumbel
        
        # 7. Indices Extraction
        ids_keep = perturbed_scores.topk(len_keep, dim=-1).indices
        
        return ids_keep

    def forward_encoder(self, x, visible_angles=None):
        """
        x: [B, 1 + N_keep, D] (CLS + visible tokens)
        visible_angles: [B, N_keep, 2] in degrees, for visible tokens only
        """
        # Compute attention bias for visible tokens (+ CLS)
        attn_bias = None
        if self.use_angular_bias and visible_angles is not None:
            attn_bias = self.enc_ang_bias(visible_angles, include_cls=True)
            # attn_bias: [B, num_heads, 1+N_keep, 1+N_keep]
        
        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)
        return self.norm(x)
    
    def forward_decoder(self, x, ids_restore, all_angles):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        cls_pe = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
        patch_pe = self.decoder_pos_embed(all_angles)
        x = x + torch.cat([cls_pe, patch_pe], dim=1)
        
        # Decoder sees ALL tokens (visible + masked), so use all angles
        attn_bias = None
        if self.use_angular_bias:
            attn_bias = self.dec_ang_bias(all_angles, include_cls=True)
        
        for blk in self.decoder_blocks:
            x = blk(x, attn_bias=attn_bias) if self.use_angular_bias else blk(x)
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
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # per-patch MSE
        loss = ((pred - target) ** 2).mean(dim=-1)        # [B, N]
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
    
    def forward(self, views, angles, mask_ratio=0.75):
        B, N, C, H, W = views.shape
        x = self.view_embed(views.view(B*N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles)
        len_keep = int(N * (1 - mask_ratio))
        
        if self.adaptive_masking and self.training:
            ids_keep = self._adaptive_masking(B, N, len_keep, views, x.device, angles=angles)
            mask = torch.ones(B, N, device=views.device, dtype=torch.bool)
            mask.scatter_(1, ids_keep, False)
            # Build ids_shuffle more robustly
            N_mask = N - len_keep
            ids_mask = mask.nonzero()[:, 1].view(B, N_mask)
            ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            mask = mask.float()
        else:
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            mask = torch.ones([B, N], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        # NEW: gather the angles of visible tokens for encoder bias
        visible_angles = torch.gather(
            angles, dim=1, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, angles.shape[2])
        )  # [B, N_keep, 2]
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x_input = torch.cat((cls_token, x_masked), dim=1)
        latent_with_cls = self.forward_encoder(x_input, visible_angles=visible_angles)
        pred = self.forward_decoder(latent_with_cls, ids_restore, angles)
        
        loss = self.forward_loss(views, pred, mask)
        return loss, pred, mask

# Model Factory
def vit_base_patch16(**kwargs):
    return PanoramicMAE(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
def vit_large_patch16(**kwargs):
    return PanoramicMAE(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
def vit_huge_patch14(**kwargs):
    return PanoramicMAE(embed_dim=1280, depth=32, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)