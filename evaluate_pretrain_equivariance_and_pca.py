import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 导入你预训练的组件 (换回 models_PanoMAE)
import util.odi_processing as odi 
from models_PanoMAE import vit_huge_patch14 

# ==========================================
# 1. 核心工具类
# ==========================================
class ERPRotator:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
        self.theta, self.phi = u * np.pi, v * np.pi / 2     
        self.xyz = torch.stack([torch.cos(self.phi)*torch.cos(self.theta), torch.cos(self.phi)*torch.sin(self.theta), torch.sin(self.phi)], dim=-1).view(-1, 3) 

    def rotate_image(self, image_np, yaw=0):
        if yaw == 0: return image_np
        img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        y_rad = np.radians(yaw)
        R = torch.tensor([[np.cos(y_rad), -np.sin(y_rad), 0], [np.sin(y_rad), np.cos(y_rad), 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        return F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def forward_patch_and_cls(self, patches, angle_centers):
        B, N, C, H, W = patches.shape
        x = self.model.view_embed(patches.view(B*N, C, H, W)).view(B, N, -1)
        x = x + self.model.angle_pos_embed(angle_centers)
        
        # 添加 CLS Token
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x_input = torch.cat((cls_token, x), dim=1)
        
        # 直接调用 Pretrain 模型内置的 forward_encoder 函数
        x_out = self.model.forward_encoder(x_input)
        
        # 返回 (Patch_Features, CLS_Token)
        return x_out[:, 1:], x_out[:, 0]

# ==========================================
# 2. 核心分析逻辑
# ==========================================
@torch.no_grad()
def run_equivariance_analysis(model, img_path, device, output_dir):
    model.eval()
    extractor = FeatureExtractor(model)
    
    # 预训练参数配置
    input_size = (1024, 2048)
    u_steps, v_steps = 32, 16
    h_fov, v_fov = 360.0/u_steps, 180.0/v_steps
    target_patch_size = 64 # 1024 / 16 = 64
    
    rotator = ERPRotator(h=input_size[0], w=input_size[1], device=device)
    os.makedirs(output_dir, exist_ok=True)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)

    # 扫描角度：0-360度，步长要和网格对齐 (360/32 = 11.25度每格)
    step_angle = 360.0 / u_steps
    test_angles = [step_angle * i for i in range(u_steps)]
    
    raw_img = np.array(Image.open(img_path).convert('RGB').resize((input_size[1], input_size[0]), Image.BICUBIC))
    
    # 0. 获取基准 (0度) 的 Patch 特征和 CLS
    patches_0 = odi.extract_patches_from_pano(raw_img, u_steps, v_steps, h_fov, v_fov, target_patch_size)
    tensors_0 = torch.stack([normalize(to_tensor(p)) for p in patches_0]).unsqueeze(0).to(device)
    base_patches, base_cls = extractor.forward_patch_and_cls(tensors_0, angle_centers)
    
    base_grid = base_patches.view(1, v_steps, u_steps, -1)

    cls_collection = []
    equiv_errors = []

    for i, ang in enumerate(test_angles):
        # 1. 物理旋转
        rot_img = rotator.rotate_image(raw_img, yaw=ang)
        
        # 2. 提取旋转后的特征
        curr_patches_raw = odi.extract_patches_from_pano(rot_img, u_steps, v_steps, h_fov, v_fov, target_patch_size)
        curr_tensors = torch.stack([normalize(to_tensor(p)) for p in curr_patches_raw]).unsqueeze(0).to(device)
        curr_patches, curr_cls = extractor.forward_patch_and_cls(curr_tensors, angle_centers)
        
        # 3. 计算 Yaw 同变性误差 (E_yaw)
        # 理论上：旋转后的特征 == 原始特征向右循环位移 i 格
        expected_grid = torch.roll(base_grid, shifts=i, dims=2) 
        actual_grid = curr_patches.view(1, v_steps, u_steps, -1)
        
        error = torch.norm(actual_grid - expected_grid) / torch.norm(base_grid)
        equiv_errors.append({'angle': ang, 'equiv_error': error.item()})
        cls_collection.append(curr_cls.squeeze().cpu().numpy())
        print(f"Angle {ang:>6.2f}° | E_yaw: {error.item():.6f}")

    # 4. 可视化 1: CLS PCA Trajectory
    cls_collection = np.array(cls_collection)
    pca = PCA(n_components=2)
    cls_2d = pca.fit_transform(cls_collection)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(cls_2d[:, 0], cls_2d[:, 1], c=test_angles, cmap='hsv', edgecolors='k')
    plt.plot(cls_2d[:, 0], cls_2d[:, 1], 'r--', alpha=0.5)
    plt.title("Pre-trained CLS Token Trajectory (Yaw 0-360°)", fontsize=14)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.colorbar(label='Rotation Angle')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "pretrain_cls_pca_trajectory_pano.png"), dpi=300)
    
    # 5. 保存数据
    pd.DataFrame(equiv_errors).to_csv(os.path.join(output_dir, "pretrain_yaw_equivariance_error_pano.csv"), index=False)
    print(f"Analysis complete. Results in {output_dir}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 实例化预训练模型 (1024x2048 -> patch_size 64)
    model = vit_huge_patch14(img_size=64, adaptive_masking=True).to(device)
    
    # 加载预训练权重
    ckpt_path = "/media/data_hdd1/shanhefu/outputs/panomae_pretrain/hybrid/panomae_pretrain_mask0.6-0.9_16*32_2e-4_epoch300_128_huge_hybrid/checkpoint-200.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    # 随便选一张你的全景图用于诊断
    run_equivariance_analysis(
        model, 
        img_path="/home/shanhefu/hybrid/test/img-7.png", # 请替换为实际的图片路径
        device=device,
        output_dir="/media/data_hdd1/shanhefu/outputs/comparison/B-2/diagnostic_pretrain/"
    )