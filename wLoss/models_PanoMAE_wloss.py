from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import math

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
    """Implements Sphere Position Embedding with optional geometric bias."""
    def __init__(self, d_model: int, num_fourier_features: int = 256, geometric_bias: bool = True):
        super().__init__()
        if d_model < 2 * num_fourier_features:
            num_fourier_features = d_model // 2
        self.num_fourier_features = num_fourier_features
        
        if geometric_bias:
            self._init_geometric_fourier_weights()
        else:
            self.register_parameter('fourier_weights', nn.Parameter(torch.randn(self.num_fourier_features, 3)))

        if 2 * self.num_fourier_features != d_model:
            self.output_proj = nn.Linear(2 * self.num_fourier_features, d_model)
        else:
            self.output_proj = nn.Identity()

    def _init_geometric_fourier_weights(self):
        """Initializes Fourier weights with a heuristic based on spherical harmonics."""
        l_max = int(math.sqrt(self.num_fourier_features))
        frequencies = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                if len(frequencies) >= self.num_fourier_features: break
                freq = torch.tensor([
                    l * math.cos(m * math.pi / (l + 1)) if l > 0 else 1.0,
                    l * math.sin(m * math.pi / (l + 1)) if l > 0 else 0.0,
                    l * 0.5 if l > 0 else 0.0
                ], dtype=torch.float32)
                frequencies.append(freq)
            if len(frequencies) >= self.num_fourier_features: break
        
        while len(frequencies) < self.num_fourier_features:
            frequencies.append(torch.randn(3))
        
        fourier_weights = torch.stack(frequencies[:self.num_fourier_features])
        self.register_parameter('fourier_weights', nn.Parameter(fourier_weights))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        lon_rad, lat_rad = torch.deg2rad(angles[..., 0]), torch.deg2rad(angles[..., 1])
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * -torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        coords_3d = torch.stack([x, y, z], dim=-1)
        p_k = torch.matmul(coords_3d, self.fourier_weights.T)
        fourier_features = torch.cat([torch.cos(p_k), torch.sin(p_k)], dim=-1)
        return self.output_proj(fourier_features)

class PanoramicMAE(nn.Module):
    """Panoramic MAE with built-in adaptive masking and Sphere Position Embedding."""
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, geometric_bias: bool = True, adaptive_masking: bool = True):
        super().__init__()
        self.view_embed = ViewEmbedder(in_chans=in_chans, embed_dim=embed_dim, img_size=img_size)
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = AnglePositionalEncoding(d_model=decoder_embed_dim, geometric_bias=geometric_bias)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (img_size**2 * in_chans), bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.adaptive_masking = adaptive_masking
        self.initialize_weights()

    def no_weight_decay(self):
        no_decay = {'mask_token'}
        no_decay.add('angle_pos_embed.fourier_weights')
        no_decay.add('decoder_pos_embed.fourier_weights')
        if isinstance(self.angle_pos_embed.output_proj, nn.Linear): no_decay.add('angle_pos_embed.output_proj.bias')
        if isinstance(self.decoder_pos_embed.output_proj, nn.Linear): no_decay.add('decoder_pos_embed.output_proj.bias')
        return no_decay
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _compute_view_importance(self, angles):
        """Computes view importance, weighting equatorial views higher."""
        lat_rad = torch.deg2rad(angles[..., 1])
        importance = torch.cos(lat_rad).abs() + 0.1 # epsilon to avoid zero importance
        return importance / importance.sum(dim=-1, keepdim=True)

    def _adaptive_masking(self, B, N, len_keep, angles, device):
        """Generates mask using importance sampling."""
        importance = self._compute_view_importance(angles)
        mask_prob = 1.0 - importance
        mask_prob /= mask_prob.sum(dim=-1, keepdim=True)
        
        masked_indices = torch.multinomial(mask_prob, N - len_keep, replacement=False)
        
        # This is a faster way to get ids_keep from masked_indices
        full_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, N, device=device, dtype=torch.bool)
        mask.scatter_(1, masked_indices, False)
        ids_keep = full_indices[mask].view(B, -1)
        
        return ids_keep

    def forward_encoder(self, x):
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

    def forward_decoder(self, x, ids_restore, all_angles):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = x_ + self.decoder_pos_embed(all_angles)
        for blk in self.decoder_blocks: x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))

    def forward_loss(self, views, pred, mask, angles):
        '''
        views: [B, N, C, H, W]
        pred: [B, N, P*P*C]
        mask: [B, N] (1 for masked, 0 for visible)
        angles: [B, N, 2] (lon, lat)
        '''
        target = views.view(views.shape[0], views.shape[1], -1)
        if self.norm_pix_loss:
            mean, var = target.mean(dim=-1, keepdim=True), target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6)**0.5
            
        # per-patch loss
        loss = ((pred - target) ** 2).mean(dim=-1) # shape: [B, N]
        
        # latitude weights
        lat_deg = angles[..., 1] # shape: [B, N]
        weights = torch.cos(torch.deg2rad(lat_deg)).abs() + 1e-6 # Add epsilon
        
        # L_wMSE: (sum(w * loss * mask) / sum(w * mask))
        weighted_loss_sum = (loss * weights * mask).sum()
        weight_sum = (weights * mask).sum()
        
        # Normalize by weight sum to keep loss magnitude stable
        return weighted_loss_sum / (weight_sum + 1e-6) # Add epsilon to denominator

    def forward(self, views, angles, mask_ratio=0.75):
        B, N, C, H, W = views.shape
        x = self.view_embed(views.view(B*N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles)
        len_keep = int(N * (1 - mask_ratio))
        
        if self.adaptive_masking:
            ids_keep = self._adaptive_masking(B, N, len_keep, angles, views.device)
            # Create mask and restore indices from ids_keep
            mask = torch.ones(B, N, device=views.device, dtype=torch.bool)
            mask.scatter_(1, ids_keep, False)
            ids_shuffle = torch.cat([ids_keep, torch.where(mask)[1].view(B, -1)], dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            mask = mask.float()
        else:
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            mask = torch.ones([B, N], device=x.device); mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        latent = self.forward_encoder(x_masked)
        pred = self.forward_decoder(latent, ids_restore, angles)
        
        # Pass angles to the loss function
        loss = self.forward_loss(views, pred, mask, angles)
        return loss, pred, mask

# Model Factory
def vit_base_patch16(**kwargs):
    return PanoramicMAE(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
def vit_large_patch16(**kwargs):
    return PanoramicMAE(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
def vit_huge_patch14(**kwargs):
    return PanoramicMAE(embed_dim=1280, depth=32, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)