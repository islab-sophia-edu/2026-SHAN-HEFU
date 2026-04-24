import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import util.odi_processing as odi
from models_segmentation_2dcnn import vit_huge_patch14 

# ==========================================
# 1. GPU 加速的旋转与指标计算工具
# ==========================================
class ERPRotatorGPU:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), 
                              torch.linspace(-1, 1, w, device=device), indexing='ij')
        self.theta, self.phi = u * np.pi, v * np.pi / 2     
        # [注] 此处底层网格构建系按照右手/标准球面投影，ERPRotator 的逻辑无需更改
        self.xyz = torch.stack([torch.cos(self.phi)*torch.cos(self.theta), 
                                torch.cos(self.phi)*torch.sin(self.theta), 
                                torch.sin(self.phi)], dim=-1).view(-1, 3) 

    def rotate(self, tensor, yaw=0, pitch=0, roll=0, mode='bilinear'):
        if yaw == 0 and pitch == 0 and roll == 0: return tensor
        
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], 
                          device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], 
                          device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], 
                          device=self.device, dtype=torch.float32)
        
        R = Rz @ Ry @ Rx
        xyz_rot = torch.matmul(self.xyz.to(torch.float32), R.T) 
        
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, 
                            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        
        return F.grid_sample(tensor, grid, mode=mode, padding_mode='border', align_corners=True)

def compute_metrics_gpu(pred, target, num_classes):
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds & target_inds).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
    return intersection, union

# ==========================================
# 2. 核心扫描逻辑
# ==========================================
@torch.no_grad()
def run_rotation_scan_gpu(model, img_dir, mask_dir, device, output_path):
    model.eval()
    pano_size = (832, 1664)
    u_steps, v_steps = 32, 16
    h_fov, v_fov = 360.0/u_steps, 180.0/v_steps
    target_patch_size = 52
    num_classes = 8

    rotator = ERPRotatorGPU(h=pano_size[0], w=pano_size[1], device=device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).unsqueeze(0).to(device)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    def rotate_angle_centers(angles_deg, yaw=0, pitch=0, roll=0):
        # 恢复左手坐标系设定
        lon = torch.deg2rad(angles_deg[..., 0])
        lat = torch.deg2rad(angles_deg[..., 1])
        
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * -torch.sin(lon) # <--- 左手坐标系
        z = torch.sin(lat)
        xyz = torch.stack([x, y, z], dim=-1)

        y_rad, p_rad, r_rad = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y_rad), -np.sin(y_rad), 0], [np.sin(y_rad), np.cos(y_rad), 0], [0, 0, 1]], device=device, dtype=torch.float32)
        Ry = torch.tensor([[np.cos(p_rad), 0, np.sin(p_rad)], [0, 1, 0], [-np.sin(p_rad), 0, np.cos(p_rad)]], device=device, dtype=torch.float32)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r_rad), -np.sin(r_rad)], [0, np.sin(r_rad), np.cos(r_rad)]], device=device, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        xyz_rot = torch.matmul(xyz, R.T)

        # 恢复左手坐标系反向计算
        lon_rot = torch.atan2(-xyz_rot[..., 1], xyz_rot[..., 0]) 
        lat_rot = torch.asin(torch.clamp(xyz_rot[..., 2], -1, 1))
        
        return torch.stack([torch.rad2deg(lon_rot), torch.rad2deg(lat_rot)], dim=-1)

    def evaluate_config(angle_cfg):
        total_inter = torch.zeros(num_classes, device=device)
        total_union = torch.zeros(num_classes, device=device)
        
        # ✅ 正确做法：测量 Rotation Stability 时，必须保持 PE 绝对固定
        current_angle_centers = angle_centers
        
        for idx, fname in enumerate(img_files):
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + '.png')
            if not os.path.exists(mask_path): continue
            
            raw_img = Image.open(img_path).convert('RGB').resize((pano_size[1], pano_size[0]), Image.BICUBIC)
            mask_pil = Image.open(mask_path).convert('L') 
            if mask_pil.size != (pano_size[1], pano_size[0]):
                mask_pil = mask_pil.resize((pano_size[1], pano_size[0]), Image.NEAREST)
            raw_mask_np = np.array(mask_pil)
            
            img_t = to_tensor(raw_img).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(raw_mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
            
            rot_img_t = rotator.rotate(img_t, **angle_cfg, mode='bilinear')
            rot_mask_t = rotator.rotate(mask_t, **angle_cfg, mode='nearest')
            
            rot_img_np = (rot_img_t.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rot_mask_np = rot_mask_t.squeeze().cpu().numpy().astype(np.uint8)
            
            # 三通道伪装法，避免 odi 工具内部广播报错
            rot_mask_3c = np.stack([rot_mask_np]*3, axis=-1)
            
            img_patches_np = odi.extract_patches_from_pano(rot_img_np, u_steps, v_steps, h_fov, v_fov, target_patch_size)
            mask_patches_3c = odi.extract_patches_from_pano(rot_mask_3c, u_steps, v_steps, h_fov, v_fov, target_patch_size)
            
            # 取回单通道标签
            mask_patches_np = [p[..., 0] for p in mask_patches_3c]
            
            patches_t = torch.stack([normalize(to_tensor(Image.fromarray(p))) for p in img_patches_np]).unsqueeze(0).to(device)
            mask_patches_t = torch.stack([torch.from_numpy(p.astype(np.int64)) for p in mask_patches_np]).unsqueeze(0).to(device)
            
            # 模型推理
            logits = model(patches_t, current_angle_centers) 
            B, N, C, Hf, Wf = logits.shape
            
            logits_up = F.interpolate(
                logits.view(B*N, C, Hf, Wf), 
                size=(target_patch_size, target_patch_size), 
                mode='bilinear', align_corners=False
            )
            pred_labels = logits_up.argmax(dim=1) 
            
            targets_flat = mask_patches_t.view(B*N, 1, mask_patches_t.shape[-2], mask_patches_t.shape[-1]).float()
            targets_resized = F.interpolate(
                targets_flat,
                size=(target_patch_size, target_patch_size),
                mode='nearest'
            ).long().squeeze(1) 
            
            inter, union = compute_metrics_gpu(pred_labels, targets_resized, num_classes)
            total_inter += inter
            total_union += union

            # 强制垃圾回收，防止 OOM
            del rot_img_t, rot_mask_t, patches_t, mask_patches_t, logits, logits_up, pred_labels, targets_flat, targets_resized
            torch.cuda.empty_cache()
            
        # 安全计算 mIoU
        valid_classes = total_union > 0
        if valid_classes.sum() > 0:
            iou_per_class = total_inter[valid_classes] / total_union[valid_classes]
            return iou_per_class.mean().item() * 100
        return 0.0

    print("Baseline (0,0,0) evaluation...")
    base_miou = evaluate_config({'yaw': 0, 'pitch': 0, 'roll': 0})
    print(f"Verified Baseline mIoU: {base_miou:.4f}%\n")

    yaw_angles = list(range(0, 360, 45))
    pitch_angles = list(range(-90, 91, 45))
    roll_angles = list(range(-180, 181, 45))

    scans = {
        'yaw':   [{'yaw': a, 'pitch': 0, 'roll': 0, 'val': a} for a in yaw_angles],
        'pitch': [{'yaw': 0, 'pitch': a, 'roll': 0, 'val': a} for a in pitch_angles],
        'roll':  [{'yaw': 0, 'pitch': 0, 'roll': a, 'val': a} for a in roll_angles],
    }

    results = []
    
    for axis, cfgs in scans.items():
        print(f"=====================================")
        print(f" Start scanning axis: {axis.upper()}")
        print(f"=====================================")
        
        for cfg in cfgs:
            val = cfg['val']
            eval_cfg = {k: v for k, v in cfg.items() if k != 'val'} 
            
            if eval_cfg['yaw'] == 0 and eval_cfg['pitch'] == 0 and eval_cfg['roll'] == 0:
                curr_miou = base_miou
                print(f"Axis: {axis.upper()} | Angle: {val:>4} (Cached Base) | mIoU: {curr_miou:.2f}%")
            else:
                curr_miou = evaluate_config(eval_cfg)
                delta = ((curr_miou - base_miou) / base_miou) * 100 if base_miou > 0 else 0
                print(f"Axis: {axis.upper()} | Angle: {val:>4} | mIoU: {curr_miou:.2f}% | Delta: {delta:+.2f}%")
                
            results.append([eval_cfg['yaw'], eval_cfg['pitch'], eval_cfg['roll'], axis, val, 
                            ((curr_miou - base_miou) / base_miou) * 100 if base_miou > 0 else 0, curr_miou])

    df = pd.DataFrame(results, columns=['yaw', 'pitch', 'roll', 'axis', 'angle', 'delta_miou_pct', 'abs_miou'])
    df.to_csv(output_path, index=False)
    print(f"\nSweep complete! Data saved to {output_path}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = vit_huge_patch14(img_size=52, patch_size=52, num_classes=8, grid_height=16, output_size=(832, 1664)).to(device)
    ckpt = torch.load("/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.6-0.9_16*32_3.2e-3_decay_huge_2dcnn_16_CE_4_view/checkpoint-99.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    run_rotation_scan_gpu(model, 
                          "/media/data_hdd2/shanhefu/CVRG-Pano/test/rgb", 
                          "/media/data_hdd2/shanhefu/CVRG-Pano/test/mask", 
                          device, 
                          "/media/data_hdd1/shanhefu/outputs/comparison/B-2/pano_seg_final_scan_gpu_fixed.csv")