import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. ERPRotator: 负责 360° 图像的 SO(3) 旋转
# ==========================================
class ERPRotator:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        # 创建坐标网格：v 从 1 到 -1 (对应纬度 90 到 -90)，u 从 -1 到 1 (对应经度 -180 到 180)
        v, u = torch.meshgrid(
            torch.linspace(1, -1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        self.theta = u * np.pi          
        self.phi = v * np.pi / 2     
        
        # 预计算 3D 笛卡尔坐标 xyz
        self.xyz = torch.stack([
            torch.cos(self.phi) * torch.cos(self.theta),
            torch.cos(self.phi) * torch.sin(self.theta),
            torch.sin(self.phi)
        ], dim=-1).view(-1, 3) 

    def _get_rotation_matrix(self, yaw, pitch, roll):
        """生成旋转矩阵 (输入为角度)"""
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        # 旋转顺序: Z(Yaw) -> Y(Pitch) -> X(Roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device, dtype=torch.float32)
        return Rz @ Ry @ Rx

    def rotate_image(self, image_np, yaw=0, pitch=0, roll=0):
        """对 numpy 图像进行 SO(3) 旋转"""
        if yaw == 0 and pitch == 0 and roll == 0:
            return image_np
        
        img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        R = self._get_rotation_matrix(yaw, pitch, roll)
        
        # 逆向映射坐标
        xyz_rotated = torch.matmul(self.xyz, R.T)
        
        # 转回球面坐标 (theta, phi)
        theta_rot = torch.atan2(xyz_rotated[:, 1], xyz_rotated[:, 0])
        phi_rot = torch.asin(torch.clamp(xyz_rotated[:, 2], -1.0, 1.0))
        
        # 转回 normalized grid [-1, 1]
        grid = torch.stack([
            theta_rot / np.pi,
            -(phi_rot / (np.pi / 2)) # 这里的负号取决于你的 v 坐标定义
        ], dim=-1).view(1, self.h, self.w, 2)
        
        rotated_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return rotated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

# ==========================================
# 2. FeatureExtractor: 适配 2D-CNN 结构的特征提取
# ==========================================
class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def forward_patch_and_cls(self, patches, angle_centers):
        """提取 Patch 特征（用于同变性）和 CLS Token（用于轨迹分析）"""
        B, N, C, H, W = patches.shape
        # 1. View Embedding (提取切面 Patch 特征)
        x = self.model.view_embed(patches.view(B*N, C, H, W)).view(B, N, -1)
        # 2. 加入球面位置编码 SPE
        x = x + self.model.angle_pos_embed(angle_centers)
        
        # 3. 运行 Transformer Encoder Blocks
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        for blk in self.model.blocks:
            x = blk(x)
        
        x = self.model.norm(x)
        
        # 分离 CLS Token 和 Patch Tokens
        cls_t = x[:, 0]        # [B, D]
        patch_t = x[:, 1:]     # [B, N, D]
        
        return patch_t, cls_t

# ==========================================
# 3. 核心算法：引入纬度权重的同变性误差计算
# ==========================================
def calculate_weighted_equiv_error(actual_patch_features, expected_patch_features, angle_centers, v_steps, u_steps):
    """
    计算基于球面面积权重(cos latitude)的同变性误差。
    actual_patch_features: [1, N, D] (旋转后模型输出的特征)
    expected_patch_features: [1, N, D] (原始特征经循环位移后的特征)
    angle_centers: [1, N, 2] (包含各 patch 的 lat/lon)
    """
    # 1. 提取每个 patch 的纬度并计算权重 cos(lat)
    # angle_centers 格式假设为 [lon, lat]
    lats_rad = torch.deg2rad(angle_centers[:, :, 1]) 
    weights = torch.cos(lats_rad).unsqueeze(-1) # [1, N, 1]
    
    # 将特征和权重重塑为网格结构 [1, V, U, D]
    actual_grid = actual_patch_features.view(1, v_steps, u_steps, -1)
    expected_grid = expected_patch_features.view(1, v_steps, u_steps, -1)
    w_grid = weights.view(1, v_steps, u_steps, 1)

    # 2. 计算加权平方差
    diff_sq = (actual_grid - expected_grid) ** 2
    base_sq = (expected_grid) ** 2
    
    # 按照球面面积进行加权求和
    weighted_diff = (diff_sq * w_grid).sum()
    weighted_base = (base_sq * w_grid).sum()
    
    # 3. 返回 RMSE 比率
    error = torch.sqrt(weighted_diff / (weighted_base + 1e-8))
    return error