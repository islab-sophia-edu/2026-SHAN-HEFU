import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import torchmetrics
import util.odi_processing as odi_helper
from models_PanoMAE import vit_huge_patch14

# ==========================================
# 1. GPU ERP 旋转器
# ==========================================
class ERPRotatorGPU:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(
            torch.linspace(1, -1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        self.theta = u * np.pi          
        self.phi = v * np.pi / 2     
        self.xyz = torch.stack([
            torch.cos(self.phi) * torch.cos(self.theta),
            torch.cos(self.phi) * torch.sin(self.theta),
            torch.sin(self.phi)
        ], dim=-1).view(-1, 3)

    def rotate(self, tensor, yaw=0, pitch=0, roll=0):
        if yaw == 0 and pitch == 0 and roll == 0: return tensor
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device, dtype=torch.float32)
        R = Rz @ Ry @ Rx
        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, 
                            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        return F.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)

# ==========================================
# 2. 标准网格采样
# ==========================================
def extract_standard_grid_patches(pano_np, h_num, v_num, patch_size):
    h, w, c = pano_np.shape
    stride_h = h // v_num
    stride_w = w // h_num
    patches = []
    for i in range(v_num):
        for j in range(h_num):
            y1, y2 = i * stride_h, (i + 1) * stride_h
            x1, x2 = j * stride_w, (j + 1) * stride_w
            patch = pano_np[y1:y2, x1:x2]
            patch_resized = np.array(Image.fromarray(patch).resize((patch_size, patch_size), Image.BICUBIC))
            patches.append(patch_resized)
    return patches

# ==========================================
# 3. 强化版 Ablation A (含 Pitch 旋转扫描)
# ==========================================
@torch.no_grad()
def run_ablation_a_with_rotation(model, img_dir, device, output_csv):
    model.eval()
    
    pano_h, pano_w = 1024, 2048
    u_steps, v_steps = 32, 16 
    patch_size = 64
    h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps
    mask_ratio = 0.75

    rotator = ERPRotatorGPU(h=pano_h, w=pano_w, device=device)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])[:20] 
    
    # 设定 Pitch 扫描角度：0度（赤道），45度，90度（极点）
    pitch_angles = [0, 15, 30, 45, 60, 75, 90]
    results = []

    for p_angle in pitch_angles:
        print(f"\n--- Testing Pitch: {p_angle} degrees ---")
        
        for fname in img_files:
            # 1. 旋转 ERP 图像
            raw_img_pil = Image.open(os.path.join(img_dir, fname)).convert('RGB').resize((pano_w, pano_h))
            img_t = torch.from_numpy(np.array(raw_img_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            rot_img_t = rotator.rotate(img_t, pitch=p_angle)
            rot_img_np = (rot_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # 2. 获取采样
            modes = {
                'Pano_Sampling': odi_helper.extract_patches_from_pano(rot_img_np, u_steps, v_steps, h_fov, v_fov, patch_size),
                'Grid_Sampling': extract_standard_grid_patches(rot_img_np, u_steps, v_steps, patch_size)
            }

            # 3. 准备角度坐标 (SPE)
            v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
            u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
            v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
            angles_t = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)

            img_results = {'name': fname, 'pitch': p_angle}

            for mode_name, patch_list in modes.items():
                psnr_metric.reset()
                gt_patches_t = torch.stack([torch.from_numpy(p).float().permute(2,0,1)/255.0 for p in patch_list]).to(device)
                views = (gt_patches_t - norm_mean) / norm_std
                views = views.unsqueeze(0)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    _, pred, mask = model(views, angles_t, mask_ratio=mask_ratio)
                
                # Tile-based Denorm 逻辑
                B, N, C, H, W = views.shape
                views_flat = views.view(B, N, -1)
                p_std = (views_flat.var(dim=-1, keepdim=True) + 1e-6).sqrt()
                p_mean = views_flat.mean(dim=-1, keepdim=True)

                pred_denorm = torch.clamp((pred.view(B, N, -1) * p_std + p_mean).view(B, N, C, H, W) * norm_std + norm_mean, 0, 1)
                views_denorm = torch.clamp(views * norm_std + norm_mean, 0, 1)

                # 计算并记录
                psnr_metric.update(pred_denorm[0], views_denorm[0])
                img_results[f"{mode_name}_PSNR"] = psnr_metric.compute().item()
            
            results.append(img_results)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # 打印汇总对比
    summary = df.groupby('pitch')[['Pano_Sampling_PSNR', 'Grid_Sampling_PSNR']].mean()
    print("\n--- Final Ablation Summary ---")
    print(summary)

if __name__ == "__main__":
    device = 'cuda'
    model = vit_huge_patch14(img_size=64, adaptive_masking=True).to(device)
    ckpt = torch.load("/media/data_hdd1/shanhefu/outputs/panomae_pretrain/hybrid/panomae_pretrain_mask0.6-0.9_16*32_2e-4_epoch300_128_huge_hybrid/checkpoint-200.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    run_ablation_a_with_rotation(
        model, 
        img_dir="/home/shanhefu/hybrid/test", 
        device=device,
        output_csv="/media/data_hdd1/shanhefu/outputs/comparison/A/ablation_a_pitch_rotation_results.csv"
    )