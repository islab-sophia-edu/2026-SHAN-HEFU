import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader

# 导入您的组件
import util.odi_processing as odi 
from models_PanoMAE_classification_rotation_easy import vit_huge_patch14 
from PanoMAE_classification_dataset_rotation_legacy import build_classification_dataset, StrictImageFolder

# ==========================================
# 0. 严格 Linear Probing (D -> C) 结构约束
# ==========================================
def enforce_linear_head(model, state_dict):
    """
    强制使用极简 Linear Head (D -> C) 进行评估。
    剥夺一切非线性拟合能力，测试真实的特征旋转不变性。
    """
    embed_dim = model.embed_dim
    num_classes = model.num_classes
    device = next(model.parameters()).device
    
    # 强制将模型 Head 设为最简 D -> C 线性映射
    model.head = nn.Linear(embed_dim, num_classes).to(device)
    
    # 安全校验：防止误加载旧的 MLP 权重
    if any(k.endswith('head.0.weight') for k in state_dict.keys()):
        print("\n[!!!] 严重警告: 你的 Checkpoint 包含 MLP Head (D->2D->C) 的权重！")
        print("[!!!] 你正在进行严格的 Linear Probing 评估，这会导致加载失败。")
        print("[!!!] 请确保你加载的是用最新 D->C 极简代码重新训练出来的 checkpoint！\n")
    else:
        print(f"\n[*] 结构校验通过: 已锁定极简 Linear Head (D={embed_dim} -> C={num_classes})")

# ==========================================
# 1. 高精度 ERP 旋转器
# ==========================================
class ERPRotator:
    def __init__(self, h, w, device='cpu'): # 默认 CPU，让 DataLoader 多进程并发执行
        self.h, self.w = h, w
        self.device = device
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), 
                              torch.linspace(-1, 1, w, device=device), indexing='ij')
        self.theta = u * np.pi          
        self.phi = v * np.pi / 2     
        self.xyz = torch.stack([torch.cos(self.phi)*torch.cos(self.theta), 
                                torch.cos(self.phi)*torch.sin(self.theta), 
                                torch.sin(self.phi)], dim=-1).view(-1, 3) 

    def rotate_image(self, image_np, yaw=0, pitch=0, roll=0):
        if yaw == 0 and pitch == 0 and roll == 0: return image_np
        img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device)
        R = (Rz @ Ry @ Rx).to(torch.float32)

        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, 
                            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        
        rotated_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        # 直接在 CPU 返回 numpy，避免 GPU-CPU 频繁通信
        return rotated_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

# ==========================================
# 2. 高性能并行扫描 Dataset
# ==========================================
class EvalSweepDataset(Dataset):
    """专门为扫描设计的 Dataset，支持多进程并发处理图像加载、旋转和切片"""
    def __init__(self, samples, transform, u_steps, v_steps, h_fov, v_fov, img_size, yaw, pitch, roll):
        self.samples = samples
        self.transform = transform
        self.u_steps = u_steps
        self.v_steps = v_steps
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.img_size = img_size
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self._rotator = None # 懒加载，防止 DataLoader 多进程初始化死锁

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._rotator is None:
            # 强制在 CPU 上运行，让 10 个 workers 并发算，完美隐藏 IO 延迟
            self._rotator = ERPRotator(512, 1024, device='cpu')
            
        path, label = self.samples[idx]
        raw_img = np.array(Image.open(path).convert('RGB').resize((1024, 512)))
        
        # 并发执行旋转
        rot_img = self._rotator.rotate_image(raw_img, yaw=self.yaw, pitch=self.pitch, roll=self.roll)
        
        # 并发执行 Patch 切割
        patches_np = odi.extract_patches_from_pano(
            pano_np=rot_img, h_num=self.u_steps, v_num=self.v_steps,
            h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=self.img_size
        )
        
        # 转换 Tensor
        patches = torch.stack([self.transform(Image.fromarray(p)) for p in patches_np])
        return patches, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. 核心扫描逻辑 (GPU Batch 推理)
# ==========================================
@torch.no_grad()
def run_classification_rotation_sweep(model, val_dataset, device, output_path, batch_size=32, num_workers=10):
    model.eval()
    
    # 获取几何参数
    v_steps = val_dataset.v_steps
    u_steps = val_dataset.u_steps
    v_fov = val_dataset.v_fov
    h_fov = val_dataset.h_fov
    img_size = val_dataset.img_size
    base_angle_centers = val_dataset.base_angles.to(device)
    
    yaw_angles = list(range(0, 360, 45))
    pitch_angles = list(range(-90, 91, 45))
    roll_angles = list(range(-180, 181, 45))

    scans = {
        'yaw':   [{'yaw': a, 'pitch': 0, 'roll': 0, 'val': a} for a in yaw_angles],
        'pitch': [{'yaw': 0, 'pitch': a, 'roll': 0, 'val': a} for a in pitch_angles],
        'roll':  [{'yaw': 0, 'pitch': 0, 'roll': a, 'val': a} for a in roll_angles],
    }
    
    samples = val_dataset.image_folder.samples
    results = []

    for axis, configs in scans.items():
        print(f"\nScanning Axis: {axis.upper()}")
        
        for cfg in configs:
            ang_val = cfg['val']
            print(f" evaluating {axis} angle: {ang_val}°... ", end="", flush=True)
            
            # 动态生成特定旋转角度的高性能 Dataset
            sweep_dataset = EvalSweepDataset(
                samples=samples, transform=val_dataset.transform,
                u_steps=u_steps, v_steps=v_steps, h_fov=h_fov, v_fov=v_fov, img_size=img_size,
                yaw=cfg['yaw'], pitch=cfg['pitch'], roll=cfg['roll']
            )
            
            # 使用 DataLoader 打包大 Batch
            loader = DataLoader(sweep_dataset, batch_size=batch_size, num_workers=num_workers, 
                                pin_memory=True, shuffle=False)
            
            correct = 0
            total = 0
            
            # GPU 极速 Batched 推理
            for patches, targets in loader:
                patches = patches.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # 动态扩展 angle_centers 匹配当前 Batch 的大小
                curr_batch_size = patches.size(0)
                batched_angles = base_angle_centers.unsqueeze(0).expand(curr_batch_size, -1, -1)
                
                with torch.cuda.amp.autocast():
                    logits = model(patches, batched_angles)
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += curr_batch_size

            acc = (correct / total) * 100
            results.append({
                'axis': axis, 'angle': ang_val, 
                'yaw': cfg['yaw'], 'pitch': cfg['pitch'], 'roll': cfg['roll'], 
                'top1_acc': acc
            })
            print(f"Accuracy: {acc:.2f}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nAll scans completed. Results saved to {output_path}")

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--batch_size', default=32, type=int) 
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--val_data_path', default='/media/data_hdd/shanhefu/sun360/sun360_outdoor_classification/test')
    parser.add_argument('--train_data_path', default='/media/data_hdd/shanhefu/sun360/sun360_outdoor_classification/train')
    parser.add_argument('--checkpoint', default='/media/data_hdd1/shanhefu/outputs/finetune/classification/huge_ft_v6_easy/checkpoint-epoch100.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    temp_folder = StrictImageFolder(root=args.train_data_path)
    full_class_to_idx = temp_folder.class_to_idx
    num_total_classes = len(temp_folder.classes) 

    model = vit_huge_patch14(
        img_size=32, 
        num_classes=num_total_classes, 
        geometric_bias=True
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # 执行严格的 Linear Probing 限制
    enforce_linear_head(model, state_dict)
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[*] 权重加载完成: {msg}")
    
    val_dataset = build_classification_dataset(
        is_train=False, 
        args=args, 
        class_to_idx=full_class_to_idx
    )

    output_csv = "/media/data_hdd1/shanhefu/outputs/comparison/FT/classification_rotation_sweep_DC.csv"
    
    # 执行高性能扫描
    run_classification_rotation_sweep(
        model, val_dataset, device, output_csv, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )