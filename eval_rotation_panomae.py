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
# 1. GPU Accelerate ERP Rotator
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
# 2. Optimized Evaluation Core
# ==========================================
@torch.no_grad()
def run_rotation_reconstruction_sweep(model, img_dir, device, output_csv):
    model.eval()
    
    pano_h, pano_w = 1024, 2048
    u_steps, v_steps = 32, 16
    patch_size = 64
    h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps
    mask_ratio = 0.75

    rotator = ERPRotatorGPU(h=pano_h, w=pano_w, device=device)
    
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])[:50]

    def evaluate_config(angle_cfg):
        psnr_metric.reset()
        ssim_metric.reset()
        
        for fname in img_files:
            raw_img_pil = Image.open(os.path.join(img_dir, fname)).convert('RGB').resize((pano_w, pano_h))
            img_t = torch.from_numpy(np.array(raw_img_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            rot_img_t = rotator.rotate(img_t, **angle_cfg)

            rot_img_np = (rot_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_patches_list = odi_helper.extract_patches_from_pano(
                rot_img_np, u_steps, v_steps, h_fov, v_fov, patch_size
            )
            
            v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
            u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
            v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
            angles_t = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)

            gt_patches_t = torch.stack([torch.from_numpy(p).float().permute(2,0,1)/255.0 for p in gt_patches_list]).to(device)
            views = (gt_patches_t - norm_mean) / norm_std
            views = views.unsqueeze(0) 

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _, pred, mask = model(views, angles_t, mask_ratio=mask_ratio)
            
            B, N, C, H, W = views.shape
            views_patches_flat = views.view(B, N, -1)
            patch_mean = views_patches_flat.mean(dim=-1, keepdim=True)
            patch_var = views_patches_flat.var(dim=-1, keepdim=True)
            patch_std = (patch_var + 1e-6).sqrt()

            pred_patches_flat = pred.view(B, N, -1)
            pred_in_global_space_flat = pred_patches_flat * patch_std + patch_mean
            pred_in_global_space = pred_in_global_space_flat.view(B, N, C, H, W)
            
            pred_denorm = torch.clamp(pred_in_global_space * norm_std + norm_mean, 0, 1)
            views_denorm = torch.clamp(views * norm_std + norm_mean, 0, 1)

            chunk_size = 64
            for i in range(0, N, chunk_size):
                chunk_pred = pred_denorm[0, i:i+chunk_size]
                chunk_gt = views_denorm[0, i:i+chunk_size]
                psnr_metric.update(chunk_pred, chunk_gt)
                ssim_metric.update(chunk_pred, chunk_gt)

            del img_t, rot_img_t, gt_patches_t, views, pred, mask
            torch.cuda.empty_cache()

        return psnr_metric.compute().item(), ssim_metric.compute().item()

    # --- Scanning Configuration ---
    print("Establishing Baseline (0,0,0)...")
    base_psnr, base_ssim = evaluate_config({'yaw': 0, 'pitch': 0, 'roll': 0})
    
    results = []
    # Updated angle ranges as requested
    scans = {
        'yaw':   [0, 45, 90, 135, 180, 225, 270, 315],
        'pitch': [-90, -45, 0, 45, 90],
        'roll':  [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    }

    for axis, angles in scans.items():
        print(f"Scanning {axis.upper()}...")
        for a in angles:
            # Skip duplicate baseline evaluation for non-yaw axis
            if a == 0 and axis != 'yaw': continue
            
            cfg = {'yaw': 0, 'pitch': 0, 'roll': 0}
            cfg[axis] = a
            
            cur_psnr, cur_ssim = evaluate_config(cfg)
            psnr_delta = ((cur_psnr - base_psnr) / base_psnr) * 100
            
            results.append({
                'axis': axis, 
                'angle': a, 
                'psnr': cur_psnr, 
                'psnr_delta_%': psnr_delta, 
                'ssim': cur_ssim
            })
            print(f"  Angle {a}: PSNR {cur_psnr:.2f} ({psnr_delta:+.2f}%)")
            
            torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    device = 'cuda'
    model = vit_huge_patch14(img_size=64, adaptive_masking=True).to(device)
    ckpt = torch.load("/media/data_hdd1/shanhefu/outputs/panomae_pretrain/hybrid/panomae_pretrain_mask0.6-0.9_16*32_2e-4_epoch300_128_huge_hybrid/checkpoint-200.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    run_rotation_reconstruction_sweep(
        model, 
        img_dir="/home/shanhefu/hybrid/test", 
        device=device,
        output_csv="/media/data_hdd1/shanhefu/outputs/comparison/B-1/panomae_reconstruction_stability.csv"
    )