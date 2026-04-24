import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import util.odi_processing as odi
import random

class Stanford2D3DDataset(Dataset):
    def __init__(self, root_dir, grid_height=16, img_size=None, is_train=True, 
                 debug_limit=None, args=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.args = args 

        if is_train:
            areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
        else:
            areas = ['area_5a', 'area_5b']

        self.filenames = []
        self.target_folder_name = 'normal' 
        
        print(f"[{'Train' if is_train else 'Val'}] Loading Stanford2D3D for Normal (using '{self.target_folder_name}')...")

        for area in areas:
            rgb_dir = os.path.join(root_dir, area, 'pano', 'rgb')
            norm_dir = os.path.join(root_dir, area, 'pano', self.target_folder_name)
            
            if not os.path.exists(rgb_dir) or not os.path.exists(norm_dir):
                continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            
            for rgb_path in rgb_files:
                file_name = os.path.basename(rgb_path)
                # 兼容 _normals.png (官方) 和 _normal.png
                norm_name = file_name.replace('_rgb.png', '_normals.png')
                norm_path = os.path.join(norm_dir, norm_name)
                
                if not os.path.exists(norm_path):
                     norm_name = file_name.replace('_rgb.png', '_normal.png')
                     norm_path = os.path.join(norm_dir, norm_name)
                
                if os.path.exists(norm_path):
                    self.filenames.append({'rgb': rgb_path, 'normal': norm_path})

        if debug_limit: self.filenames = self.filenames[:debug_limit]
        print(f"Found {len(self.filenames)} pairs.")
        if len(self.filenames) == 0:
            raise RuntimeError(f"No valid image pairs found in {root_dir}.")

        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        # 预计算角度中心 (Flattened)
        # 注意：这里生成的顺序必须和 extract_patches_from_pano 的提取顺序一致
        # odi_processing 通常是两层循环: for phi (v) ... for theta (u) ...
        
        # 纬度 (Latitude, Phi): 从 +90 到 -90
        v_centers = torch.linspace(90 - self.v_fov/2, -90 + self.v_fov/2, self.v_steps)
        # 经度 (Longitude, Theta): 从 -180 到 180
        u_centers = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
        
        # shape: (N_patches, 2) -> [lon, lat]
        self.base_angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1) 

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.filenames[idx]
        
        try:
            img_pil = Image.open(data['rgb']).convert('RGB')
            norm_pil = Image.open(data['normal']).convert('RGB')

            if self.args and hasattr(self.args, 'pano_h'):
                target_h, target_w = self.args.pano_h, self.args.pano_w
                if img_pil.size != (target_w, target_h):
                    img_pil = img_pil.resize((target_w, target_h), Image.BICUBIC)
                    norm_pil = norm_pil.resize((target_w, target_h), Image.NEAREST)

            img_np = np.array(img_pil)
            norm_raw = np.array(norm_pil).astype(np.float32)

            # 1. 映射 [0, 255] -> [-1, 1]
            norm_np = (norm_raw / 255.0) * 2.0 - 1.0
            
            # 2. 验证并强制单位向量
            norm_len = np.linalg.norm(norm_np, axis=2, keepdims=True)
            norm_np = norm_np / (norm_len + 1e-8)
            
            # 3. 生成 Mask (RGB内容 & 法线模长合理性)
            valid_mask = (img_np.sum(axis=2) > 10) & (norm_len.squeeze() > 0.1) & (norm_len.squeeze() < 1.7)
            valid_mask = valid_mask.astype(np.float32)

            # [修复核心] 4. 处理数据增强与角度同步
            current_angles = self.base_angle_centers.clone()

            # 检查 args 是否启用 horizontal_roll
            use_roll = self.is_train and getattr(self.args, 'use_horizontal_roll', False)

            if use_roll:
                W = img_np.shape[1]
                roll_idx = random.randint(0, W - 1)
                
                # A. 滚动图像内容
                img_np = np.roll(img_np, roll_idx, axis=1)
                norm_np = np.roll(norm_np, roll_idx, axis=1)
                valid_mask = np.roll(valid_mask, roll_idx, axis=1)
                
                # B. 同步修正角度 (Positional Encoding)
                # 图像向右滚了 roll_idx，相当于视口向左移了，或者说原来的 (0,0) 变成了新的位置
                # 计算滚动的角度量
                roll_deg = (roll_idx / W) * 360.0
                
                # 图像内容是 wrap around 的。
                # 如果图像 roll 了 +shift，原本在 index 0 的内容跑到了 index shift。
                # 我们的 extract_patches 是按固定网格切分的。
                # 切分出来的第 i 个 patch，它的内容变了（来自原图的左边区域）。
                # 所以它的 经度 (Longitude) 应该减去 roll_deg。
                
                current_angles[:, 0] = current_angles[:, 0] - roll_deg
                
                # 归一化到 [-180, 180]
                # 简单的办法: ((angle + 180) % 360) - 180
                current_angles[:, 0] = ((current_angles[:, 0] + 180) % 360) - 180

            # Extraction
            raw_h = img_np.shape[0] // self.v_steps
            raw_w = img_np.shape[1] // self.u_steps
            target_size = (int(raw_w), int(raw_h)) if raw_h != raw_w else int(raw_h)

            rgb_patches_list = odi.extract_patches_from_pano(
                pano_np=img_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            norm_patches_list = odi.extract_patches_from_pano(
                pano_np=norm_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            mask_patches_list = odi.extract_patches_from_pano(
                pano_np=valid_mask, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            rgb_tensors = []
            norm_tensors = []
            mask_tensors = []
            
            for i in range(len(rgb_patches_list)):
                p_rgb = Image.fromarray(rgb_patches_list[i])
                p_rgb_tensor = self.normalize(transforms.ToTensor()(p_rgb))
                rgb_tensors.append(p_rgb_tensor)

                p_norm_np = norm_patches_list[i] 
                p_norm_tensor = torch.from_numpy(p_norm_np).permute(2, 0, 1).float()
                norm_tensors.append(p_norm_tensor)
                
                p_mask_np = mask_patches_list[i]
                mask_tensors.append(torch.from_numpy(p_mask_np).float().unsqueeze(0))

            patches_img = torch.stack(rgb_tensors)    
            patches_norm = torch.stack(norm_tensors) 
            patches_mask = torch.stack(mask_tensors)

        except Exception as e:
            print(f"[Error] {e}")
            h_fb = 52
            w_fb = 52
            if self.args:
                h_fb = self.args.pano_h // self.v_steps
                w_fb = self.args.pano_w // self.u_steps
            return (torch.zeros(self.u_steps * self.v_steps, 3, h_fb, w_fb),
                    self.base_angle_centers, # Fallback
                    torch.zeros(self.u_steps * self.v_steps, 3, h_fb, w_fb),
                    torch.zeros(self.u_steps * self.v_steps, 1, h_fb, w_fb))

        return patches_img, current_angles, patches_norm, patches_mask

def build_normal_dataset(is_train, args):
    return Stanford2D3DDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=None,
        is_train=is_train,
        debug_limit=None,
        args=args
    )