import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import torchmetrics
from torchvision.utils import save_image

# 仅导入 PanoMAE
import util.odi_processing as odi_helper
from models_PanoMAE import vit_huge_patch14 as pano_vit_huge

# ==========================================
# 1. 核心工具类
# ==========================================
class ERPRotatorGPU:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
        self.theta, self.phi = u * np.pi, v * np.pi / 2     
        self.xyz = torch.stack([torch.cos(self.phi)*torch.cos(self.theta), torch.cos(self.phi)*torch.sin(self.theta), torch.sin(self.phi)], dim=-1).view(-1, 3) 

    def rotate(self, tensor, yaw=0, pitch=0, roll=0, mode='bilinear'):
        if yaw == 0 and pitch == 0 and roll == 0: return tensor
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device, dtype=torch.float32)
        R = Rz @ Ry @ Rx
        xyz_rot = torch.matmul(self.xyz.to(torch.float32), R.T) 
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        return F.grid_sample(tensor, grid, mode=mode, padding_mode='border', align_corners=True)

class LatitudeStratifiedEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.bands = {
            'Band1_NorthPole': (0.0, 0.15),
            'Band2_NorthHigh': (0.15, 0.35),
            'Band3_Equator':   (0.35, 0.65),
            'Band4_SouthHigh': (0.65, 0.85),
            'Band5_SouthPole': (0.85, 1.0)
        }
        self.metrics = {k: {
            'psnr': torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device),
            'ssim': torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            'l2': torchmetrics.MeanSquaredError().to(device)
        } for k in self.bands}

    def reset(self):
        for k in self.bands:
            for m in self.metrics[k].values(): m.reset()

    def update(self, pred_pano, gt_pano):
        # 确保输入是 [B, C, H, W] 且 H = 512 (已被外部统一 Resize)
        H = pred_pano.shape[2]
        for band_name, (start_pct, end_pct) in self.bands.items():
            start_y, end_y = int(start_pct * H), int(end_pct * H)
            p_band, g_band = pred_pano[:, :, start_y:end_y, :], gt_pano[:, :, start_y:end_y, :]
            
            # 使用 chunk 更新防止 VRAM 溢出
            self.metrics[band_name]['psnr'].update(p_band, g_band)
            self.metrics[band_name]['ssim'].update(p_band, g_band)
            self.metrics[band_name]['l2'].update(p_band.reshape(-1), g_band.reshape(-1))

    def compute(self, prefix=''):
        res = {}
        for k in self.bands:
            res[f'{prefix}{k}_PSNR'] = self.metrics[k]['psnr'].compute().item()
            res[f'{prefix}{k}_SSIM'] = self.metrics[k]['ssim'].compute().item()
            res[f'{prefix}{k}_L2']   = self.metrics[k]['l2'].compute().item()
        return res

# ==========================================
# 2. PanoMAE 专属逻辑
# ==========================================
@torch.no_grad()
def run_panomae_evaluation(model, img_dir, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 原始推理分辨率
    pano_h, pano_w = 1024, 2048
    # 评估与保存的目标统一分辨率
    target_h, target_w = 512, 1024
    
    u_steps, v_steps = 32, 16
    patch_size = 64
    h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps
    mask_ratio = 0.75

    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    evaluator = LatitudeStratifiedEvaluator(device=device)

    # SPE 坐标
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angles_t = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)
    angles_gpu = angles_t[0]

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])[:100]
    
    # ---------------- Phase 1: 扫描并计算 Band 误差 ----------------
    print(f"Phase 1: PanoMAE Scanning {len(img_files)} images for Latitude Error...")
    
    image_scores = []
    
    for idx, fname in enumerate(img_files):
        img_path = os.path.join(img_dir, fname)
        raw_img_pil = Image.open(img_path).convert('RGB').resize((pano_w, pano_h))
        img_np = np.array(raw_img_pil)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            gt_patches_list = odi_helper.extract_patches_from_pano(img_np, u_steps, v_steps, h_fov, v_fov, patch_size)
            pano_views = (torch.stack([torch.from_numpy(p).float().permute(2,0,1)/255.0 for p in gt_patches_list]).to(device) - norm_mean) / norm_std
            pano_views = pano_views.unsqueeze(0)
            
            _, p_pred, _ = model(pano_views, angles_t, mask_ratio=mask_ratio)
            
            B, N, C, H, W = pano_views.shape
            p_flat = pano_views.view(B, N, -1)
            p_std, p_mean = (p_flat.var(dim=-1, keepdim=True) + 1e-6).sqrt(), p_flat.mean(dim=-1, keepdim=True)
            p_pred_denorm = torch.clamp((p_pred.view(B, N, -1) * p_std + p_mean).view(B, N, C, H, W) * norm_std + norm_mean, 0, 1)
            p_views_denorm = torch.clamp(pano_views * norm_std + norm_mean, 0, 1)
            
            # 拼回 1024x2048
            pano_rec_erp_1024 = odi_helper.embed_patches_to_pano_gpu(p_pred_denorm[0], pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu).unsqueeze(0)
            gt_erp_1024 = odi_helper.embed_patches_to_pano_gpu(p_views_denorm[0], pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu).unsqueeze(0)
            
            # 【核心操作】将结果 Resize 到 512x1024，以便与 Baseline 对齐评估
            pano_rec_erp_512 = F.interpolate(pano_rec_erp_1024, size=(target_h, target_w), mode='bilinear', align_corners=False)
            gt_erp_512 = F.interpolate(gt_erp_1024, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            evaluator.update(pano_rec_erp_512, gt_erp_512)
            
            # 快速计算一个整体 PSNR 用于后续选图
            psnr_val = torchmetrics.functional.peak_signal_noise_ratio(pano_rec_erp_512, gt_erp_512, data_range=1.0).item()
            image_scores.append({'fname': fname, 'pano_psnr': psnr_val})

        if (idx + 1) % 10 == 0: print(f"  Processed {idx + 1}/{len(img_files)}")

    # 保存 PanoMAE 的 Band 结果
    pano_band_results = evaluator.compute(prefix='Pano_')
    pd.DataFrame([pano_band_results]).to_csv(os.path.join(output_dir, "panomae_latitude_error.csv"), index=False)
    pd.DataFrame(image_scores).to_csv(os.path.join(output_dir, "panomae_image_scores.csv"), index=False)
    print("\n[✔] PanoMAE Latitude Error & Scores Saved.")

    # ---------------- Phase 2: 旋转与可视化生成 ----------------
    # 选出表现最差的（或者最好的，这里挑最差的来体现 Baseline 更差）前 5 张
    top_5_names = [x['fname'] for x in sorted(image_scores, key=lambda x: x['pano_psnr'], reverse=True)[:10]]
    print(f"\nPhase 2: Generating PanoMAE visualizations for Top 5 Samples: {top_5_names}")

    rot_configs = [
        {'name': 'NoRot', 'y': 0, 'p': 0, 'r': 0},
        {'name': 'Pitch60', 'y': 0, 'p': 60, 'r': 0},
        {'name': 'Yaw90', 'y': 90, 'p': 0, 'r': 0},
        {'name': 'Roll45', 'y': 0, 'p': 0, 'r': 45}
    ]
    rotator = ERPRotatorGPU(h=pano_h, w=pano_w, device=device)

    for fname in top_5_names:
        img_np = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB').resize((pano_w, pano_h)))
        save_folder = os.path.join(output_dir, fname.split('.')[0])
        os.makedirs(save_folder, exist_ok=True)
        
        # 保存一张基准的 512x1024 图像
        Image.fromarray(img_np).resize((target_w, target_h)).save(os.path.join(save_folder, "00_original_512.png"))

        for cfg in rot_configs:
            img_t = torch.from_numpy(img_np).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            rot_img_t = rotator.rotate(img_t, yaw=cfg['y'], pitch=cfg['p'], roll=cfg['r'])
            rot_img_np = (rot_img_t.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            gt_patches_list = odi_helper.extract_patches_from_pano(rot_img_np, u_steps, v_steps, h_fov, v_fov, patch_size)
            pano_views = (torch.stack([torch.from_numpy(p).float().permute(2,0,1)/255.0 for p in gt_patches_list]).to(device) - norm_mean) / norm_std
            pano_views = pano_views.unsqueeze(0) 

            # 推理 (注意不旋转 SPE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _, p_pred, p_mask = model(pano_views, angles_t, mask_ratio=mask_ratio)
            
            p_flat = pano_views.view(1, u_steps*v_steps, -1)
            p_std, p_mean = (p_flat.var(dim=-1, keepdim=True) + 1e-6).sqrt(), p_flat.mean(dim=-1, keepdim=True)
            p_pred_denorm = torch.clamp((p_pred.view(1, u_steps*v_steps, -1) * p_std + p_mean).view(1, u_steps*v_steps, 3, 64, 64) * norm_std + norm_mean, 0, 1)
            p_views_denorm = torch.clamp(pano_views * norm_std + norm_mean, 0, 1)

            p_mask_exp = p_mask[0].view(u_steps*v_steps, 1, 1, 1)
            
            pano_pure_rec = odi_helper.embed_patches_to_pano_gpu(p_pred_denorm[0], pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu).unsqueeze(0)

            # 统一 Resize 到 512x1024
            rot_img_512 = F.interpolate(rot_img_t, size=(target_h, target_w), mode='bilinear')
            pano_pure_512 = F.interpolate(pano_pure_rec, size=(target_h, target_w), mode='bilinear')

            sf = cfg['name']
            save_image(rot_img_512[0], os.path.join(save_folder, f"1_GT_{sf}.png"))
            save_image(pano_pure_512[0], os.path.join(save_folder, f"2_PanoMAE_Predict_{sf}.png"))
            
    print(f"\n[✔] PanoMAE visuals generated.")

if __name__ == "__main__":
    device = 'cuda'
    model_pano = pano_vit_huge(img_size=64, adaptive_masking=True).to(device)
    ckpt_pano = torch.load("/media/data_hdd1/shanhefu/outputs/panomae_pretrain/hybrid/panomae_pretrain_mask0.6-0.9_16*32_2e-4_epoch300_128_huge_hybrid/checkpoint-200.pth", map_location=device, weights_only=False)
    model_pano.load_state_dict(ckpt_pano['model'])
    
    run_panomae_evaluation(
        model_pano, 
        img_dir="/media/data_hdd/shanhefu/sun360/sun360_outdoor/test", 
        device=device,
        output_dir="/media/data_hdd1/shanhefu/outputs/comparison/pretrain_visual_gap"
    )