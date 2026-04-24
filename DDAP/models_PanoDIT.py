# models_panodit.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_PanoMAE import ViewEmbedder, AnglePositionalEncoding

# ==========================================
# Module 2: DiT Blocks
# ==========================================
class AdaLN(nn.Module):
    def __init__(self, hidden_size, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * hidden_size, bias=True)
        )
        # AdaLN-Zero trick
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, condition):
        chunks = self.adaLN_modulation(condition).chunk(6, dim=-1)
        scale_sa, shift_sa, gate_sa, scale_mlp, shift_mlp, gate_mlp = chunks
        return scale_sa, shift_sa, gate_sa, scale_mlp, shift_mlp, gate_mlp

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, condition_dim=None, text_dim=768):
        super().__init__()
        condition_dim = condition_dim or hidden_size
        self.adaLN = AdaLN(hidden_size, condition_dim)
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size) # cross-attn pre-norm
        
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, kdim=text_dim, vdim=text_dim, batch_first=True)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
    
    def forward(self, x, c_timestep, text_feat):
        scale_sa, shift_sa, gate_sa, scale_mlp, shift_mlp, gate_mlp = self.adaLN(x, c_timestep)
        
        # 1. Self-Attention (AdaLN modulated)
        x_norm = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * attn_out
        
        # 2. Cross-Attention (Text)
        x_norm3 = self.norm3(x)
        cross_out, _ = self.cross_attn(query=x_norm3, key=text_feat, value=text_feat)
        x = x + cross_out
        
        # 3. MLP (AdaLN modulated)
        x_norm_mlp = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm_mlp)
        return x

# ==========================================
# Module 3: Embedders
# ==========================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.freq_dim = freq_dim
    
    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)

class TextEmbedder(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def forward(self, text_list):
        device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(text_list, padding="max_length", truncation=True, 
                                return_tensors="pt", max_length=77).to(device)
        return self.text_encoder(**tokens).last_hidden_state

# ==========================================
# Module 5: Latent Tangent Extractor
# ==========================================
class LatentTangentExtractor(nn.Module):
    """
    在 VAE Latent Grid 上进行切平面提取（无梯度操作）。
    自包含逻辑，适应 Latent 空间的通道数和尺寸。
    """
    def __init__(self, grid_height=4, latent_h=64, latent_w=128, latent_patch_size=8, device='cuda'):
        super().__init__()
        self.v_num = grid_height
        self.u_num = 2 * grid_height
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patch_size = latent_patch_size
        self.device = torch.device(device)
        
        self.angles_rad = self._generate_angles()
        self.angles_deg = torch.from_numpy(np.degrees(self.angles_rad)).float()
        self.h_fov_deg = 360.0 / self.u_num
        self.v_fov_deg = 180.0 / self.v_num

        # 预计算 patch 网格，加速推理
        P = self.patch_size
        u_lin = torch.linspace(-(P - 1) / 2.0, (P - 1) / 2.0, P)
        v_lin = torch.linspace((P - 1) / 2.0, -(P - 1) / 2.0, P)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing='xy')
        self.register_buffer('_pixel_u', uu.reshape(-1))
        self.register_buffer('_pixel_v', vv.reshape(-1))

    def _generate_angles(self):
        v_fov_rad = np.radians(180.0 / self.v_num)
        phis = np.linspace(np.pi/2 - v_fov_rad/2, -np.pi/2 + v_fov_rad/2, self.v_num)
        thetas = np.linspace(-np.pi, np.pi, self.u_num, endpoint=False)
        grid_phis, grid_thetas = np.meshgrid(phis, thetas, indexing='ij')
        return np.stack([grid_thetas.flatten(), grid_phis.flatten()], axis=1)

    def extract(self, latent):
        B, C, H, W = latent.shape
        N = self.angles_deg.shape[0]
        P = self.patch_size
        
        angles_t = self.angles_deg.to(self.device)
        theta = torch.deg2rad(angles_t[:, 0])
        phi = torch.deg2rad(angles_t[:, 1])

        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        nc = torch.stack([cos_phi*cos_theta, -cos_phi*sin_theta, sin_phi], dim=1)
        xn = torch.stack([-sin_theta, -cos_theta, torch.zeros_like(theta)], dim=1)
        yn = torch.stack([-sin_phi*cos_theta, sin_phi*sin_theta, cos_phi], dim=1)

        L = (P / 2.0) / torch.tan(torch.tensor(np.radians(self.h_fov_deg) / 2.0, device=self.device))

        uu = self._pixel_u
        vv = self._pixel_v

        pts = (uu.view(1, -1, 1) * xn.unsqueeze(1) +
               vv.view(1, -1, 1) * yn.unsqueeze(1) +
               L * nc.unsqueeze(1))

        px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
        p_norm = torch.sqrt(px**2 + py**2 + pz**2 + 1e-8)
        theta_odi = torch.atan2(-py, px)
        phi_odi = torch.asin(torch.clamp(pz / p_norm, -1.0, 1.0))

        grid_x = theta_odi / torch.pi
        grid_y = -phi_odi / (torch.pi / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, P, P, 2)

        # 扩展 grid 以处理 batch
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B*N, P, P, 2)
        latent_exp = latent.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B*N, C, H, W)
        
        patches = F.grid_sample(latent_exp, grid, mode='bicubic', padding_mode='border', align_corners=True)
        patches = patches.view(B, N, C, P, P)
        return patches, self.angles_deg

# ==========================================
# Module 4: 核心 PanoDiT 模型
# ==========================================
class PanoDiT(nn.Module):
    def __init__(self, latent_channels=4, latent_patch_size=8, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0, text_dim=768, grid_height=4):
        super().__init__()
        
        # ✅ 复用 PanoMAE 的特征提取与球面编码
        self.view_embed = ViewEmbedder(in_chans=latent_channels, embed_dim=embed_dim, img_size=latent_patch_size)
        self.angle_pos_embed = AnglePositionalEncoding(d_model=embed_dim, geometric_bias=True)
        
        # Conditioning
        self.timestep_embed = TimestepEmbedder(hidden_size=embed_dim)
        
        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                     condition_dim=embed_dim, text_dim=text_dim)
            for _ in range(depth)
        ])
        
        # Output Projection (预测噪声)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, latent_channels * latent_patch_size * latent_patch_size)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        
        self.grid_height = grid_height
        self.latent_patch_size = latent_patch_size
        self.latent_channels = latent_channels

    def forward(self, noisy_latent_patches, angles, timestep, text_feat):
        B, N, C, P, _ = noisy_latent_patches.shape
        
        # 1. View Embedding
        x = self.view_embed(noisy_latent_patches.view(B*N, C, P, P)).view(B, N, -1)
        # 2. Spherical Positional Encoding
        x = x + self.angle_pos_embed(angles)
        # 3. Timestep Embedding
        t_emb = self.timestep_embed(timestep)
        
        # 4. Forward DiT Blocks
        for block in self.blocks:
            x = block(x, c_timestep=t_emb, text_feat=text_feat)
            
        # 5. Output
        x = self.final_proj(self.final_norm(x))
        pred_noise = x.view(B, N, C, P, P)
        return pred_noise

    def load_from_panomae_checkpoint(self, checkpoint_path):
        """精准迁移权重，处理 In_Chans 从 3 变为 4 的情况"""
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt)
        compatible_keys = {}
        
        for k, v in state_dict.items():
            # 处理 ViewEmbedder 通道扩增
            if k == 'view_embed.proj.proj.weight':
                new_v = torch.zeros_like(self.view_embed.proj.proj.weight)
                # 复用RGB通道，Alpha/Latent的第4通道初始化为0
                new_v[:, :3, :, :] = v
                compatible_keys[k] = new_v
            
            elif k.startswith('angle_pos_embed'):
                compatible_keys[k] = v
                
            elif k.startswith('blocks.'):
                # 映射 PanoMAE (timm.Block) 到 DiTBlock
                if 'norm1' in k:
                    compatible_keys[k] = v
                elif 'attn.qkv' in k:
                    # 分解 qkv 给 nn.MultiheadAttention
                    # nn.MultiheadAttention 内部是 in_proj_weight
                    pass # 严格映射较复杂，建议随机初始化自注意力，或在此处细写映射逻辑。为求稳定，允许部分缺失重新学习
                elif 'norm2' in k:
                    compatible_keys[k] = v
                elif 'mlp.fc1' in k:
                    compatible_keys[k.replace('mlp.fc1', 'mlp.0')] = v
                elif 'mlp.fc2' in k:
                    compatible_keys[k.replace('mlp.fc2', 'mlp.2')] = v

        missing, unexpected = self.load_state_dict(compatible_keys, strict=False)
        print(f"Loaded from PanoMAE. Transferred keys: {len(compatible_keys)}")
        print(f"Missing (randomly initialized for DiT, e.g. adaLN, cross_attn, qkv): {len(missing)}")