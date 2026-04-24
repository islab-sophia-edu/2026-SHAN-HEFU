import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer

# 继承 timm 的 VisionTransformer，确保兼容性
class BaselineVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=None, **kwargs):
        
        # [关键修复] 显式调用父类，且不传递 qk_scale (因为它已被弃用)
        # 这里的 kwargs 会吸收掉所有多余的参数，防止报错
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            num_classes=num_classes, embed_dim=embed_dim, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs
        )
        # 移除 head (如果是纯分割提取特征用，或者保留做分类)
        # 这里为了分割任务，我们通常输出 feature map 或者直接输出 logits
        # 你的 Baseline 看起来是直接输出 (B, C, H, W) 的 Logits？
        # 如果是直接分类 ViT，输出是 (B, num_classes)。
        # 如果是分割 ViT (如 Segmenter)，需要 Decoder。
        
        # 假设你用的是最简单的 ViT + Conv Head 进行分割 (根据你之前的代码逻辑)
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape
        
        # 1. Patch Embedding & Encoder
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        # Blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # 2. Reshape back to Image (Remove CLS)
        x = x[:, 1:, :] # (B, N, D)
        
        # 计算 Grid Size
        grid_h = H // self.patch_embed.patch_size[0]
        grid_w = W // self.patch_embed.patch_size[1]
        
        x = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        
        # 3. Simple Segmentation Head
        x = self.head(x) # (B, num_classes, gh, gw)
        
        # 4. Upsample to Input Size
        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

# 工厂函数 (Factory Functions)
# [关键修复] 使用 **kwargs 接收 patch_size，避免与默认值冲突
def vit_base_patch16(**kwargs):
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model_kwargs.update(kwargs) # 更新用户传入的 patch_size 等
    if 'patch_size' not in model_kwargs: model_kwargs['patch_size'] = 16
    return BaselineVisionTransformer(**model_kwargs)

def vit_large_patch16(**kwargs):
    model_kwargs = dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model_kwargs.update(kwargs)
    if 'patch_size' not in model_kwargs: model_kwargs['patch_size'] = 16
    return BaselineVisionTransformer(**model_kwargs)

def vit_huge_patch14(**kwargs):
    model_kwargs = dict(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model_kwargs.update(kwargs)
    # 如果用户传了 patch_size=32，这里就不会覆盖它
    if 'patch_size' not in model_kwargs: model_kwargs['patch_size'] = 14
    return BaselineVisionTransformer(**model_kwargs)