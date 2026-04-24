import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF

import util.odi_processing

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
        
        rotated_tensor = F.grid_sample(img_tensor, grid, mode='bicubic', padding_mode='border', align_corners=True)
        return rotated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

class StrictImageFolder(ImageFolder):
    """
    1. 自动过滤 'others' 类别。
    2. 强制执行特定的 class-to-index 映射，解决不同数据集之间的类别不对齐问题。
    """
    def __init__(self, root, predefined_class_to_idx: Optional[Dict[str, int]] = None, **kwargs):
        self.predefined_class_to_idx = predefined_class_to_idx
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        if 'others' in classes:
            classes.remove('others')
            
        if self.predefined_class_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx
        else:
            valid_classes = [c for c in classes if c in self.predefined_class_to_idx]
            class_to_idx = {c: self.predefined_class_to_idx[c] for c in valid_classes}
            return valid_classes, class_to_idx


class PanoClassificationDataset(Dataset):
    def __init__(self, root_dir, grid_height=4, img_size=128, transform=None, is_train=True, device='cuda', class_to_idx=None):
        self.root_dir = root_dir
        self.image_folder = StrictImageFolder(root=root_dir, predefined_class_to_idx=class_to_idx)
        self.class_to_idx = self.image_folder.class_to_idx
        self.classes = self.image_folder.classes
        self.is_train = is_train
        
        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.img_size = img_size
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        v_step_size = 180.0 / self.v_steps
        v_centers_deg = torch.linspace(90 - v_step_size / 2, -90 + v_step_size / 2, self.v_steps)
        u_centers_deg = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers_deg, u_centers_deg, indexing='ij')
        
        self.base_angles = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)
        self.transform = transform
        
        self._rotator = None  # 不在这里初始化
        self._pano_h = None
        self._pano_w = None
        
        # 验证集初始化为 'none'
        self.aug_stage = 'yaw_only' if is_train else 'none'

    def set_aug_stage(self, stage: str):
        """由 main.py 调用，用于动态切换 Curriculum Learning 阶段"""
        # 【修复安全锁】只有训练集允许被修改增强阶段，保护验证集！
        if self.is_train:
            self.aug_stage = stage

    def __len__(self):
        return len(self.image_folder)

    def _get_rotator(self, h, w):
        # 每个worker第一次调用时才初始化，强制用CPU
        if self._rotator is None or self._pano_h != h or self._pano_w != w:
            self._rotator = ERPRotator(h, w, device='cpu')
            self._pano_h, self._pano_w = h, w
        return self._rotator

    def __getitem__(self, idx):
        path, label = self.image_folder.samples[idx]
        
        yaw_shift, pitch_shift, roll_shift = 0.0, 0.0, 0.0
        if self.aug_stage in ['yaw_only', 'full']:
            yaw_shift = random.uniform(0, 360)
        if self.aug_stage == 'full':
            pitch_shift = random.uniform(-30.0, 30.0)
            roll_shift  = random.uniform(-30.0, 30.0)

        try:
            with open(path, 'rb') as f:
                pano_pil = Image.open(f).convert('RGB')
                pano_np = np.array(pano_pil)  # uint8

            # 先旋转全景图，再提取Patch
            if yaw_shift != 0 or pitch_shift != 0 or roll_shift != 0:
                rotator = self._get_rotator(pano_np.shape[0], pano_np.shape[1])
                pano_np = rotator.rotate_image(pano_np, yaw=yaw_shift, 
                                               pitch=pitch_shift, roll=roll_shift)

            patches_np_list = util.odi_processing.extract_patches_from_pano(
                pano_np=pano_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=self.img_size
            )

            if self.transform:
                patches = torch.stack([self.transform(Image.fromarray(p)) for p in patches_np_list])
            else:
                patches = torch.stack([transforms.ToTensor()(p) for p in patches_np_list])

        except Exception as e:
            print(f"Warning: Failed to load {path}, Error: {e}")
            patches = torch.zeros(self.u_steps * self.v_steps, 3, self.img_size, self.img_size)

        # 角度标签永远用base_angles（不随旋转变化）
        return patches, self.base_angles.clone(), torch.tensor(label, dtype=torch.long)


def build_classification_dataset(is_train, args, class_to_idx=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dynamic_img_size = 512 // args.grid_height

    if is_train:
        t = [
            # 1. 强化版颜色抖动 (亮度、对比度、饱和度、色相)
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            
            # 2. 随机灰度化 (20%概率，极大地提升对光照和色彩的泛化能力)
            transforms.RandomGrayscale(p=0.2),
            
            # 3. 随机高斯模糊 (20%概率，模拟相机失焦或低分辨率，提升鲁棒性)
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ]
        # 4. 随机擦除 / Cutout (类似 Crop 的局部遮挡效果)
        if hasattr(args, 'reprob') and args.reprob > 0:
            t.append(transforms.RandomErasing(p=args.reprob, value='random'))
            
        transform = transforms.Compose(t)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        
    return PanoClassificationDataset(
        root_dir=args.train_data_path if is_train else args.val_data_path,
        grid_height=args.grid_height, img_size=dynamic_img_size, transform=transform,
        is_train=is_train, device=args.device, class_to_idx=class_to_idx
    )