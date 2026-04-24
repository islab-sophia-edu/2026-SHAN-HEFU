import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchmetrics
import util.odi_processing as odi 
from models_segmentation_2dcnn import vit_huge_patch14 

# ==========================================
# 1. 配置与颜色表
# ==========================================
PALETTE = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
], dtype=np.uint8)

# ==========================================
# 2. 旋转器
# ==========================================
class ERPRotator:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
        theta, phi = u * np.pi, v * np.pi / 2     
        self.xyz = torch.stack([torch.cos(phi)*torch.cos(theta), torch.cos(phi)*torch.sin(theta), torch.sin(phi)], dim=-1).view(-1, 3) 

    def _get_rotation_matrix(self, yaw, pitch, roll):
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device)
        return (Rz @ Ry @ Rx).float()

    def rotate(self, data_np, yaw=0, pitch=0, roll=0, is_label=False):
        mode = 'nearest' if is_label else 'bilinear'
        data_t = torch.from_numpy(data_np).float().to(self.device)
        if not is_label: data_t = data_t.permute(2, 0, 1).unsqueeze(0)
        else: data_t = data_t.unsqueeze(0).unsqueeze(0)

        R = self._get_rotation_matrix(yaw, pitch, roll)
        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, 
                            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        
        output = F.grid_sample(data_t, grid, mode=mode, padding_mode='border', align_corners=True)
        if is_label: return output.squeeze().cpu().numpy().astype(np.uint8)
        return output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

# ==========================================
# 3. GPU 拼接工具
# ==========================================
def embed_patches_to_pano_gpu(patches_tensor, pano_h, pano_w, h_num, v_num, h_fov_deg, v_fov_deg, angles):
    device = patches_tensor.device
    N, C, H_patch, W_patch = patches_tensor.shape
    pano_canvas = torch.zeros((C, pano_h, pano_w), device=device, dtype=torch.float32)
    
    y_range = torch.arange(pano_h, device=device, dtype=torch.float32)
    x_range = torch.arange(pano_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    theta_omni = (2.0 * grid_x / (pano_w - 1) - 1.0) * torch.pi 
    phi_omni = (0.5 - grid_y / (pano_h - 1)) * torch.pi         
    
    cos_phi = torch.cos(phi_omni)
    pn = torch.stack([cos_phi * torch.cos(theta_omni), -cos_phi * torch.sin(theta_omni), torch.sin(phi_omni)], dim=0)
    L = (W_patch / 2.0) / torch.tan(torch.deg2rad(torch.tensor(h_fov_deg / 2.0, device=device)))
    
    for i in range(N):
        theta_rad = torch.deg2rad(angles[i, 0])
        phi_rad = torch.deg2rad(angles[i, 1])
        nc = torch.tensor([torch.cos(phi_rad) * torch.cos(theta_rad), -torch.cos(phi_rad) * torch.sin(theta_rad), torch.sin(phi_rad)], device=device).view(3, 1, 1)
        xn = torch.tensor([-torch.sin(theta_rad), -torch.cos(theta_rad), 0.0], device=device).view(3, 1, 1)
        yn = torch.tensor([-torch.sin(phi_rad) * torch.cos(theta_rad), torch.sin(phi_rad) * torch.sin(theta_rad), torch.cos(phi_rad)], device=device).view(3, 1, 1)
        
        cos_alpha = torch.sum(pn * nc, dim=0) 
        threshold = 2 * L / torch.sqrt(torch.tensor(W_patch**2 + H_patch**2 + 4*L**2, device=device))
        mask_fov = cos_alpha >= threshold
        
        if not mask_fov.any(): continue
        
        r_dist = L / (cos_alpha + 1e-8) 
        xp, yp = r_dist * torch.sum(pn * xn, dim=0), r_dist * torch.sum(pn * yn, dim=0)
        
        mask_rect = (xp > -W_patch/2.0) & (xp < W_patch/2.0) & (yp > -H_patch/2.0) & (yp < H_patch/2.0)
        final_mask = mask_fov & mask_rect
        if not final_mask.any(): continue
        
        c1 = torch.clamp((W_patch / 2.0 + xp[final_mask] - 0.5).round().long(), 0, W_patch - 1)
        r1 = torch.clamp((H_patch / 2.0 - yp[final_mask] - 0.5).round().long(), 0, H_patch - 1)
        
        pano_canvas[:, final_mask] = patches_tensor[i, :, r1, c1] 

    return pano_canvas 

# ==========================================
# 4. 核心评估与可视化筛选 (仅 PanoMAE)
# ==========================================
@torch.no_grad()
def evaluate_and_visualize_panomae(model, img_dir, gt_dir, device, output_root):
    model.eval()
    rotator = ERPRotator(832, 1664, device)
    u_steps, v_steps, p_size = 32, 16, 52
    h_fov, v_fov = 360/u_steps, 180/v_steps
    
    miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=8).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 【核心修正】基础相机网格 - 全程使用
    vc = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    uc = torch.linspace(-180, 180, u_steps + 1)[:-1]
    vg, ug = torch.meshgrid(vc, uc, indexing='ij')
    base_pano_angles = torch.stack([ug.flatten(), vg.flatten()], dim=1).to(device)

    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])[:50]
    results_log = []

    print("Phase 1: Scanning for Top 5 samples where PanoMAE performs best under rotation...")
    for fname in files:
        img_path = os.path.join(img_dir, fname)
        gt_path = os.path.join(gt_dir, fname.replace(".jpg", ".png"))
        if not os.path.exists(gt_path): continue

        img = np.array(Image.open(img_path).convert('RGB').resize((1664, 832)))
        gt = np.array(Image.open(gt_path).resize((1664, 832), Image.NEAREST))
        
        test_rots = [{'y': 0, 'p': 60, 'r': 0}, {'y': 90, 'p': 0, 'r': 0}, {'y': 0, 'p': 0, 'r': 45}]
        avg_miou = 0

        for r_cfg in test_rots:
            r_img = rotator.rotate(img, yaw=r_cfg['y'], pitch=r_cfg['p'], roll=r_cfg['r'])
            r_gt = rotator.rotate(gt, yaw=r_cfg['y'], pitch=r_cfg['p'], roll=r_cfg['r'], is_label=True)
            r_gt_t = torch.from_numpy(r_gt).long().to(device)

            p_patches = torch.stack([normalize(transforms.ToTensor()(p)) for p in odi.extract_patches_from_pano(r_img, u_steps, v_steps, h_fov, v_fov, p_size)]).unsqueeze(0).to(device)
            
            # 【核心修正】模型推理只使用 base_pano_angles
            p_idx = torch.argmax(model(p_patches, base_pano_angles.unsqueeze(0)), dim=2).squeeze(0)
            
            # 画布拼接同样使用 base_pano_angles
            p_mask_gpu = embed_patches_to_pano_gpu(p_idx.unsqueeze(1).float(), 832, 1664, u_steps, v_steps, h_fov, v_fov, base_pano_angles)
            p_mask_t = p_mask_gpu.squeeze(0).long()
            
            avg_miou += miou_metric(p_mask_t, r_gt_t).item()

        results_log.append({'name': fname, 'avg_miou': avg_miou / 3})

    top_5 = sorted(results_log, key=lambda x: x['avg_miou'], reverse=True)[:5]
    top_names = [x['name'] for x in top_5]
    print(f"Top 5 Samples found: {top_names}")

    # Phase 2: 详细旋转可视化
    vis_scenarios = [
        {'n': 'Yaw_90', 'y': 90, 'p': 0, 'r': 0},
        {'n': 'Pitch_45', 'y': 0, 'p': 45, 'r': 0},
        {'n': 'Pitch_60', 'y': 0, 'p': 60, 'r': 0},
        {'n': 'Pitch_90', 'y': 0, 'p': 90, 'r': 0},
        {'n': 'Roll_45', 'y': 0, 'p': 0, 'r': 45}
    ]

    for fname in top_names:
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB').resize((1664, 832)))
        gt = np.array(Image.open(os.path.join(gt_dir, fname.replace(".jpg", ".png"))).resize((1664, 832), Image.NEAREST))
        
        save_path = os.path.join(output_root, fname.split('.')[0])
        os.makedirs(save_path, exist_ok=True)
        Image.fromarray(img).save(os.path.join(save_path, "00_ORIGINAL.png"))

        for sc in vis_scenarios:
            r_img = rotator.rotate(img, yaw=sc['y'], pitch=sc['p'], roll=sc['r'])
            r_gt = rotator.rotate(gt, yaw=sc['y'], pitch=sc['p'], roll=sc['r'], is_label=True)
            
            p_patches = torch.stack([normalize(transforms.ToTensor()(p)) for p in odi.extract_patches_from_pano(r_img, u_steps, v_steps, h_fov, v_fov, p_size)]).unsqueeze(0).to(device)
            
            # 【核心修正】推理只使用 base_pano_angles
            p_idx = torch.argmax(model(p_patches, base_pano_angles.unsqueeze(0)), dim=2).squeeze(0)
            
            # 【核心修正】拼接只使用 base_pano_angles
            p_mask_gpu = embed_patches_to_pano_gpu(p_idx.unsqueeze(1).float(), 832, 1664, u_steps, v_steps, h_fov, v_fov, base_pano_angles)
            p_mask_np = p_mask_gpu.squeeze(0).cpu().numpy().astype(np.int32)
            
            # 着色并保存
            p_vis = PALETTE[p_mask_np]
            suffix = sc['n']
            Image.fromarray(r_img).save(os.path.join(save_path, f"{suffix}_1_ROT_IMG.png"))
            Image.fromarray(PALETTE[r_gt]).save(os.path.join(save_path, f"{suffix}_2_GT.png"))
            Image.fromarray(p_vis).save(os.path.join(save_path, f"{suffix}_3_PanoMAE_Pred.png"))

    print(f"Visualization complete. Saved to: {output_root}")

if __name__ == "__main__":
    device = 'cuda'
    model = vit_huge_patch14(img_size=52, patch_size=52, num_classes=8, grid_height=16).to(device)
    ckpt = torch.load("/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.6-0.9_16*32_3.2e-3_decay_huge_2dcnn_16_CE_4_view/checkpoint-99.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    evaluate_and_visualize_panomae(
        model, 
        img_dir="/media/data_hdd2/shanhefu/CVRG-Pano/test/rgb", 
        gt_dir="/media/data_hdd2/shanhefu/CVRG-Pano/test/mask",
        device=device,
        output_root="/media/data_hdd1/shanhefu/outputs/comparison/B-2/panomae_best_samples/"
    )