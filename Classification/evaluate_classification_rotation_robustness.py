import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import argparse

# 导入您的组件
import util.odi_processing as odi 
from models_PanoMAE_classification import vit_huge_patch14 
from PanoMAE_classification_dataset import build_classification_dataset, StrictImageFolder

# ==========================================
# 1. 高精度 ERP 旋转器 (升级支持 Yaw, Pitch, Roll)
# ==========================================
class ERPRotator:
    def __init__(self, h, w, device='cuda'):
        self.h, self.w = h, w
        self.device = device
        # 预计算坐标网格
        v, u = torch.meshgrid(torch.linspace(1, -1, h, device=device), 
                              torch.linspace(-1, 1, w, device=device), indexing='ij')
        self.theta = u * np.pi          
        self.phi = v * np.pi / 2     
        self.xyz = torch.stack([torch.cos(self.phi)*torch.cos(self.theta), 
                                torch.cos(self.phi)*torch.sin(self.theta), 
                                torch.sin(self.phi)], dim=-1).view(-1, 3) 

    def rotate_image(self, image_np, yaw=0, pitch=0, roll=0):
        if yaw == 0 and pitch == 0 and roll == 0: return image_np
        
        # 将图片转为 Tensor 并移至设备
        img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 计算 3D 旋转矩阵
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], device=self.device)
        Ry = torch.tensor([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], device=self.device)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], device=self.device)
        R = (Rz @ Ry @ Rx).to(torch.float32)

        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi, 
                            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1, 1)) / (np.pi/2))], dim=-1).view(1, self.h, self.w, 2)
        
        # 执行双线性插值旋转
        rotated_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return rotated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

# ==========================================
# 2. 核心扫描逻辑
# ==========================================
@torch.no_grad()
def run_classification_rotation_sweep(model, val_dataset, device, output_path):
    model.eval()
    
    # 获取几何参数
    v_steps = val_dataset.v_steps
    u_steps = val_dataset.u_steps
    v_fov = val_dataset.v_fov
    h_fov = val_dataset.h_fov
    img_size = val_dataset.img_size
    angle_centers = val_dataset.angle_centers.unsqueeze(0).to(device)

    # 初始化旋转器
    rotator = ERPRotator(h=512, w=1024, device=device)
    
    # 定义测试角度范围
    yaw_angles = list(range(0, 360, 45))
    pitch_angles = list(range(-90, 91, 45))
    roll_angles = list(range(-180, 181, 45))

    scans = {
        'yaw':   [{'yaw': a, 'pitch': 0, 'roll': 0, 'val': a} for a in yaw_angles],
        'pitch': [{'yaw': 0, 'pitch': a, 'roll': 0, 'val': a} for a in pitch_angles],
        'roll':  [{'yaw': 0, 'pitch': 0, 'roll': a, 'val': a} for a in roll_angles],
    }
    
    samples = val_dataset.image_folder.samples
    num_samples = len(samples)
    results = []

    # 遍历不同的扫描轴 (Yaw, Pitch, Roll)
    for axis, configs in scans.items():
        print(f"\nScanning Axis: {axis.upper()}")
        
        for cfg in configs:
            ang_val = cfg['val']
            print(f" Evaluating {axis} angle: {ang_val}°...")
            correct = 0
            total = 0
            
            for i in range(num_samples):
                path, label = samples[i]
                
                # 1. 加载并旋转
                raw_img = np.array(Image.open(path).convert('RGB').resize((1024, 512)))
                # 传入 yaw, pitch, roll 配置
                rot_img = rotator.rotate_image(raw_img, yaw=cfg['yaw'], pitch=cfg['pitch'], roll=cfg['roll'])
                
                # 2. 提取 Patch
                patches_np = odi.extract_patches_from_pano(
                    pano_np=rot_img, h_num=u_steps, v_num=v_steps,
                    h_fov_deg=h_fov, v_fov_deg=v_fov, out_size=img_size
                )
                
                # 3. 预处理并移至设备
                patches = torch.stack([val_dataset.transform(Image.fromarray(p)) for p in patches_np]).unsqueeze(0).to(device)
                target_label = torch.tensor(label, device=device).unsqueeze(0)
                
                # 4. 推理
                logits = model(patches, angle_centers)
                pred = torch.argmax(logits, dim=1)
                
                if pred.item() == target_label.item():
                    correct += 1
                total += 1

            acc = (correct / total) * 100
            # 记录详细结果
            results.append({
                'axis': axis, 
                'angle': ang_val, 
                'yaw': cfg['yaw'], 
                'pitch': cfg['pitch'], 
                'roll': cfg['roll'], 
                'top1_acc': acc
            })
            print(f" --> Accuracy: {acc:.2f}%")

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nAll scans completed. Results saved to {output_path}")

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--val_data_path', default='/media/data_hdd/shanhefu/sun360/sun360_outdoor_classification/test')
    parser.add_argument('--train_data_path', default='/media/data_hdd/shanhefu/sun360/sun360_outdoor_classification/train')
    parser.add_argument('--checkpoint', default='/media/data_hdd1/shanhefu/outputs/finetune/classification/classification_mask0.6-0.9_16*32_8e-3_decay_huge_512/checkpoint-best.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    temp_folder = StrictImageFolder(root=args.train_data_path)
    full_class_to_idx = temp_folder.class_to_idx
    num_total_classes = len(temp_folder.classes) 

    # 实例化模型
    model = vit_huge_patch14(
        img_size=32, 
        num_classes=num_total_classes, 
        geometric_bias=True
    ).to(device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    # 构建验证数据集
    val_dataset = build_classification_dataset(
        is_train=False, 
        args=args, 
        class_to_idx=full_class_to_idx
    )

    # 执行扫描
    output_csv = "/media/data_hdd1/shanhefu/outputs/comparison/B-2/classification_rotation_sweep.csv"
    run_classification_rotation_sweep(model, val_dataset, device, output_csv)