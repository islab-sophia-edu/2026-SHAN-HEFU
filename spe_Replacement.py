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
# 1. GPU ERP Rotator
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
        self.theta, self.phi = u * np.pi, v * np.pi / 2     
        self.xyz = torch.stack([
            torch.cos(self.phi) * torch.cos(self.theta),
            torch.cos(self.phi) * torch.sin(self.theta),
            torch.sin(self.phi)
        ], dim=-1).view(-1, 3)

    def rotate(self, tensor, yaw=0, pitch=0, roll=0):
        if yaw == 0 and pitch == 0 and roll == 0: return tensor
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device)
        R = (Rz @ Ry @ Rx).float()
        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, 
                            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        return F.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)

# ==========================================
# 2. Ablation B Core Logic (SPE vs 2D PE)
# ==========================================
@torch.no_grad()
def run_ablation_b_full_scan(model, img_dir, device, output_csv):
    model.eval()
    
    pano_h, pano_w = 1024, 2048
    u_steps, v_steps, patch_size = 32, 16, 64
    h_fov, v_fov = 360.0/u_steps, 180.0/v_steps
    mask_ratio = 0.75

    rotator = ERPRotatorGPU(h=pano_h, w=pano_w, device=device)
    
    # Metrics setup
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])[:50]

    # --- 1. SPE Grid (Spherical) ---
    vc = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    uc = torch.linspace(-180, 180, u_steps + 1)[:-1]
    vg, ug = torch.meshgrid(vc, uc, indexing='ij')
    spe_angles = torch.stack([ug.flatten(), vg.flatten()], dim=1).unsqueeze(0).to(device)

    # --- 2. 2D PE Grid (Linear Mapping) ---
    v_2d = torch.linspace(90, -90, v_steps)
    u_2d = torch.linspace(-180, 180, u_steps)
    vg_2d, ug_2d = torch.meshgrid(v_2d, u_2d, indexing='ij')
    pe_2d_angles = torch.stack([ug_2d.flatten(), vg_2d.flatten()], dim=1).unsqueeze(0).to(device)

    def evaluate_with_pos_config(angle_cfg, mode='SPE'):
        psnr_metric.reset()
        ssim_metric.reset()
        for fname in img_files:
            img_pil = Image.open(os.path.join(img_dir, fname)).convert('RGB').resize((pano_w, pano_h))
            img_t = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            rot_img_t = rotator.rotate(img_t, **angle_cfg)
            rot_np = (rot_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            patches_list = odi_helper.extract_patches_from_pano(rot_np, u_steps, v_steps, h_fov, v_fov, patch_size)
            gt_t = torch.stack([torch.from_numpy(p).float().permute(2,0,1)/255.0 for p in patches_list]).to(device)
            views = (gt_t - norm_mean) / norm_std
            views = views.unsqueeze(0)

            input_angles = spe_angles if mode == 'SPE' else pe_2d_angles

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _, pred, _ = model(views, input_angles, mask_ratio=mask_ratio)
            
            v_flat = views.view(1, views.shape[1], -1)
            p_std, p_mean = (v_flat.var(dim=-1, keepdim=True) + 1e-6).sqrt(), v_flat.mean(dim=-1, keepdim=True)
            pred_denorm = torch.clamp((pred.view(1, views.shape[1], -1) * p_std + p_mean).view(views.shape) * norm_std + norm_mean, 0, 1)
            gt_denorm = torch.clamp(views * norm_std + norm_mean, 0, 1)
            
            psnr_metric.update(pred_denorm[0], gt_denorm[0])
            ssim_metric.update(pred_denorm[0], gt_denorm[0])

            del img_t, rot_img_t, views, pred, pred_denorm, gt_denorm
            torch.cuda.empty_cache()
            
        return psnr_metric.compute().item(), ssim_metric.compute().item()

    # --- Scanning Angles Configuration ---
    yaw_angles = list(range(0, 360, 45))
    pitch_angles = list(range(-90, 91, 45))
    roll_angles = list(range(-180, 181, 45))
    
    scans = {
        'yaw':   [{'yaw': a, 'pitch': 0, 'roll': 0, 'val': a} for a in yaw_angles],
        'pitch': [{'yaw': 0, 'pitch': a, 'roll': 0, 'val': a} for a in pitch_angles],
        'roll':  [{'yaw': 0, 'pitch': 0, 'roll': a, 'val': a} for a in roll_angles],
    }

    results = []
    
    for axis, configs in scans.items():
        print(f"\nScanning Axis: {axis.upper()}")
        for cfg in configs:
            ang_val = cfg.pop('val')
            print(f"  Evaluating {axis} at {ang_val} degrees...")
            
            p_spe, s_spe = evaluate_with_pos_config(cfg, mode='SPE')
            p_2d, s_2d = evaluate_with_pos_config(cfg, mode='2DPE')
            
            results.append({
                'axis': axis,
                'angle': ang_val,
                'psnr_spe': p_spe,
                'psnr_2dpe': p_2d,
                'psnr_gap': p_spe - p_2d,
                'ssim_spe': s_spe,
                'ssim_2dpe': s_2d,
                'ssim_gap': s_spe - s_2d
            })
            print(f"    [PSNR] SPE: {p_spe:.2f} | 2DPE: {p_2d:.2f} | Gap: {p_spe-p_2d:.2f}")
            print(f"    [SSIM] SPE: {s_spe:.4f} | 2DPE: {s_2d:.4f} | Gap: {s_spe-s_2d:.4f}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nAblation B Full Scan (PSNR & SSIM) saved to: {output_csv}")

if __name__ == "__main__":
    device = 'cuda'
    model = vit_huge_patch14(img_size=64, adaptive_masking=True).to(device)
    ckpt = torch.load("/media/data_hdd1/shanhefu/outputs/panomae_pretrain/hybrid/panomae_pretrain_mask0.6-0.9_16*32_2e-4_epoch300_128_huge_hybrid/checkpoint-200.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    run_ablation_b_full_scan(model, "/home/shanhefu/hybrid/test", device, 
                               "/media/data_hdd1/shanhefu/outputs/comparison/A/ablation_b_spe_vs_2dpe_full.csv")