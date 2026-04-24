import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 导入你提供的组件
import util.odi_processing as odi 
from models_segmentation_2dcnn import vit_huge_patch14 

# ==========================================
# 1. 颜色表 (CVRG-Pano 8类) - 用于可视化
# ==========================================
PALETTE = np.array([
    [255, 0, 0],    # Wall
    [0, 255, 0],    # Floor
    [0, 0, 255],    # Cabinet
    [255, 255, 0],  # Bed
    [255, 0, 255],  # Chair
    [0, 255, 255],  # Sofa
    [128, 0, 0],    # Table
    [0, 128, 0],    # Door
], dtype=np.uint8)

# ==========================================
# 2. 核心工具类
# ==========================================
class ERPRotator:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
        self.theta = u * np.pi          
        self.phi = v * np.pi / 2     
        self.xyz = torch.stack([torch.cos(self.phi)*torch.cos(self.theta), torch.cos(self.phi)*torch.sin(self.theta), torch.sin(self.phi)], dim=-1).view(-1, 3) 

    def _get_rotation_matrix(self, yaw, pitch, roll):
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device, dtype=torch.float32)
        return Rz @ Ry @ Rx

    def rotate_image(self, image_np, yaw=0, pitch=0, roll=0):
        if yaw == 0 and pitch == 0 and roll == 0: return image_np
        img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        R = self._get_rotation_matrix(yaw, pitch, roll)
        xyz_rotated = torch.matmul(self.xyz, R.T)
        theta_rot, phi_rot = torch.atan2(xyz_rotated[:, 1], xyz_rotated[:, 0]), torch.asin(torch.clamp(xyz_rotated[:, 2], -1.0, 1.0))
        grid = torch.stack([theta_rot / np.pi, -(phi_rot / (np.pi / 2))], dim=-1).view(1, self.h, self.w, 2)
        rotated_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return rotated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def get_cls_token(self, patches, angle_centers):
        B, N, C, H, W = patches.shape
        # 1. View Embedding
        x = self.model.view_embed(patches.view(B*N, C, H, W)).view(B, N, -1)
        # 2. Positional Embedding
        x = x + self.model.angle_pos_embed(angle_centers)
        # 3. Add CLS Token
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 4. Encoder Blocks
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 0] # 返回 CLS Token [B, D]

# ==========================================
# 3. 运行评估与可视化
# ==========================================
@torch.no_grad()
def run_cls_sim_and_vis(model, img_dir, device, output_dir):
    model.eval()
    extractor = FeatureExtractor(model)
    rotator = ERPRotator(h=832, w=1664, device=device)
    os.makedirs(output_dir, exist_ok=True)
    
    input_size = (832, 1664)
    u_steps, v_steps = 32, 16
    h_fov, v_fov = 360.0/u_steps, 180.0/v_steps
    target_patch_size = 52 
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    # 预计算角度中心 (采样网格)
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)

    # 扫描角度 (按照你的建议，细化扫描)
    # 这里我们以 Pitch 扫描为例，因为它最能体现几何鲁棒性
    test_angles = [0, 30, 60, 90] 
    
    # 选择一张测试图片进行可视化
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    target_fname = img_files[0]
    img_path = os.path.join(img_dir, target_fname)
    raw_img = np.array(Image.open(img_path).convert('RGB').resize((input_size[1], input_size[0]), Image.BICUBIC))
    
    sim_results = []
    
    # 0. 基准特征提取 (0度)
    patches_0 = odi.extract_patches_from_pano(raw_img, u_steps, v_steps, h_fov, v_fov, target_patch_size)
    tensors_0 = torch.stack([normalize(to_tensor(p)) for p in patches_0]).unsqueeze(0).to(device)
    base_cls = extractor.get_cls_token(tensors_0, angle_centers)
    
    for ang in test_angles:
        print(f"Processing Pitch Angle: {ang}...")
        
        # 1. 物理旋转
        rot_img = rotator.rotate_image(raw_img, yaw=0, pitch=ang, roll=0)
        
        # 2. 从旋转后的图中提取 Patch 并获取 CLS Token (B-2-2)
        curr_patches = odi.extract_patches_from_pano(rot_img, u_steps, v_steps, h_fov, v_fov, target_patch_size)
        curr_tensors = torch.stack([normalize(to_tensor(p)) for p in curr_patches]).unsqueeze(0).to(device)
        curr_cls = extractor.get_cls_token(curr_tensors, angle_centers)
        
        # 计算余弦相似度
        sim = F.cosine_similarity(base_cls, curr_cls).item()
        sim_results.append({'pitch_angle': ang, 'cls_similarity': sim})
        
        # 3. 分割结果可视化 B-2-3 (选择 0, 30, 60 度)
        if ang in [0, 30, 60]:
            logits = model(curr_tensors, angle_centers) # [1, N, C, Ph, Pw]
            preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # [N, Ph, Pw]
            
            # 类别 ID 映射到颜色
            color_patches = []
            for i in range(preds.shape[0]):
                # 将 ID 矩阵 preds[i] 直接索引 PALETTE 得到 RGB 矩阵
                color_patch = PALETTE[preds[i]] # Shape: [Ph, Pw, 3]
                color_patches.append(color_patch)
            
            # 使用 ODI 的拼接工具还原回全景格式
            # 注意：embed_patches_to_pano 通常返回 0-1 的 float 数组
            full_mask_vis = odi.embed_patches_to_pano(
                color_patches, pano_size=input_size, 
                h_num=u_steps, v_num=v_steps, h_fov_deg=h_fov, v_fov_deg=v_fov
            )
            
            # 保存旋转后的原图和对应的分割图，方便论文对比
            Image.fromarray(rot_img).save(os.path.join(output_dir, f"raw_rot_pitch_{ang}.png"))
            vis_pil = Image.fromarray((full_mask_vis * 255).astype(np.uint8))
            vis_pil.save(os.path.join(output_dir, f"vis_mask_pitch_{ang}.png"))

    # 保存相似度 CSV
    pd.DataFrame(sim_results).to_csv(os.path.join(output_dir, "cls_similarity_pitch.csv"), index=False)
    print(f"Results saved to {output_dir}. Task complete.")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 实例化模型 (根据你的 Huge 2D-CNN 配置)
    model = vit_huge_patch14(img_size=52, patch_size=52, num_classes=8, grid_height=16).to(device)
    
    # 加载权重
    ckpt_path = "/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.6-0.9_16*32_3.2e-3_decay_huge_2dcnn_16_CE_4_view/checkpoint-99.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    run_cls_sim_and_vis(
        model, 
        img_dir="/home/shanhefu/CVRG-Pano/test/rgb", 
        device=device,
        output_dir="/media/data_hdd1/shanhefu/outputs/comparison/B-2/vis_and_sim/"
    )