from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import math

# Reuse Dataset's logic logic or Component
class ViewEmbedder(nn.Module):
    def __init__(self, in_chans=6, embed_dim=768, img_size=224):
        super().__init__()
        from timm.models.layers import PatchEmbed
        # in_chans is now 6
        self.proj = PatchEmbed(
            img_size=img_size, patch_size=img_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )

    def forward(self, x):
        return self.proj(x)

class AnglePositionalEncoding(nn.Module):
    """Standard Sphere PE"""
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
        l_max = int(math.sqrt(self.num_fourier_features))
        frequencies = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                if len(frequencies) >= self.num_fourier_features: break
                freq = torch.tensor([l * math.cos(m * math.pi / (l + 1)) if l > 0 else 1.0, l * math.sin(m * math.pi / (l + 1)) if l > 0 else 0.0, l * 0.5 if l > 0 else 0.0], dtype=torch.float32)
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

class PanoramicMAE6CH(nn.Module):
    """
    Geometry-Aware PanoMAE with 6-channel input (RGB + XYZ).
    Reconstructs both Texture (RGB) and Geometry (XYZ).
    """
    def __init__(self, img_size=224, in_chans=6, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, geometric_bias: bool = True, adaptive_masking: bool = True):
        super().__init__()
        
        self.in_chans = in_chans # 6
        self.patch_size = img_size # Treating whole view as patch
        
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_embed = ViewEmbedder(in_chans=in_chans, embed_dim=embed_dim, img_size=img_size)
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=geometric_bias)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = AnglePositionalEncoding(d_model=decoder_embed_dim, geometric_bias=geometric_bias)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Prediction Head: Output 6 channels (RGB + XYZ) per pixel
        self.decoder_pred = nn.Linear(decoder_embed_dim, (img_size**2 * in_chans), bias=True)
        
        self.norm_pix_loss = norm_pix_loss
        self.adaptive_masking = adaptive_masking
        self.initialize_weights()

    def no_weight_decay(self):
        no_decay = {'mask_token', 'cls_token'}
        no_decay.add('angle_pos_embed.fourier_weights')
        no_decay.add('decoder_pos_embed.fourier_weights')
        if isinstance(self.angle_pos_embed.output_proj, nn.Linear): no_decay.add('angle_pos_embed.output_proj.bias')
        if isinstance(self.decoder_pos_embed.output_proj, nn.Linear): no_decay.add('decoder_pos_embed.output_proj.bias')
        return no_decay
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        imgs: (B, N, 6, H, W)
        x: (B, N, L*6)
        """
        B, N, C, H, W = imgs.shape
        # Flatten patches: (B, N, 6*H*W)
        x = imgs.reshape(B, N, C * H * W) 
        return x

    def unpatchify(self, x):
        """
        x: (B, N, L*6)
        imgs: (B, N, 6, H, W)
        """
        p = self.patch_size
        c = self.in_chans
        B, N, L = x.shape
        imgs = x.reshape(B, N, c, p, p)
        return imgs

    def _compute_view_importance(self, angles):
        lat_rad = torch.deg2rad(angles[..., 1])
        importance = torch.cos(lat_rad).abs() + 0.1 
        return importance / importance.sum(dim=-1, keepdim=True)

    def _adaptive_masking(self, B, N, len_keep, angles, device):
        importance = self._compute_view_importance(angles)
        mask_prob = 1.0 - importance
        mask_prob /= mask_prob.sum(dim=-1, keepdim=True)
        masked_indices = torch.multinomial(mask_prob, N - len_keep, replacement=False)
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

    def forward_loss(self, views, pred, mask):
        """
        views: [B, N, 6, H, W]
        pred: [B, N, 6*H*W]
        mask: [B, N]
        """
        # 1. 获取 Patch 像素数 (用于拆分)
        # pred shape: [B, N, 6 * patch_h * patch_w]
        # 我们假设通道顺序是 RGBXYZ，所以直接对半切分即可
        
        target = self.patchify(views) # [B, N, 6*H*W]
        
        dim = target.shape[-1] // 2  # 一半是RGB，一半是XYZ
        
        target_rgb = target[:, :, :dim]
        target_xyz = target[:, :, dim:]
        
        pred_rgb = pred[:, :, :dim]
        pred_xyz = pred[:, :, dim:]

        # 2. 分别计算 MSE Loss
        loss_rgb = (pred_rgb - target_rgb) ** 2
        loss_xyz = (pred_xyz - target_xyz) ** 2
        
        loss_rgb = loss_rgb.mean(dim=-1) # [B, N]
        loss_xyz = loss_xyz.mean(dim=-1) # [B, N]
        
        # 3. 只计算被 Mask 掉的部分
        loss_rgb = (loss_rgb * mask).sum() / (mask.sum() + 1e-6)
        loss_xyz = (loss_xyz * mask).sum() / (mask.sum() + 1e-6)
        
        # 4. 总 Loss (直接相加)
        total_loss = loss_rgb + loss_xyz
        
        # [修改] 返回三个值
        return total_loss, loss_rgb, loss_xyz

    def forward(self, views, angles, mask_ratio=0.75):
        # views: (B, N, 6, H, W)
        B, N, C, H, W = views.shape
        
        # Embed
        x = self.view_embed(views.view(B*N, C, H, W)).view(B, N, -1)
        x = x + self.angle_pos_embed(angles)
        
        # Masking
        len_keep = int(N * (1 - mask_ratio))
        if self.adaptive_masking:
            ids_keep = self._adaptive_masking(B, N, len_keep, angles, views.device)
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

        # Encoder
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        cls_token = self.cls_token.expand(B, -1, -1)
        x_input = torch.cat((cls_token, x_masked), dim=1)
        latent_with_cls = self.forward_encoder(x_input)
        latent = latent_with_cls[:, 1:, :]

        # Decoder
        pred = self.forward_decoder(latent, ids_restore, angles) # Output (B, N, 6*H*W)
        
        # Loss
        loss, loss_rgb, loss_xyz = self.forward_loss(views, pred, mask)
        
        # [修改] 返回 5 个值
        return loss, pred, mask, loss_rgb, loss_xyz

# Factory
def vit_base_patch16(**kwargs):
    # in_chans=6 default
    return PanoramicMAE6CH(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=6, **kwargs)
def vit_large_patch16(**kwargs):
    return PanoramicMAE6CH(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=6, **kwargs)
def vit_huge_patch14(**kwargs):
    return PanoramicMAE6CH(embed_dim=1280, depth=32, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=6, **kwargs)