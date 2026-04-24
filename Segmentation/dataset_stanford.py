import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import util.odi_processing as odi
import random

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from torchvision import transforms
import util.odi_processing as odi
import random

class Stanford2D3DDataset(Dataset):
    def __init__(self, root_dir, grid_height=16, img_size=None, is_train=True, 
                 num_classes=13, debug_limit=None, args=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.num_classes = num_classes
        self.args = args 

        if is_train:
            areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
        else:
            areas = ['area_5a', 'area_5b']

        self.filenames = []
        self.mask_folder_name = 'anno' # 确认文件夹名，如果是 'anno' 请改回
        
        print(f"[{'Train' if is_train else 'Val'}] Loading Stanford2D3D Segmentation...")

        for area in areas:
            rgb_dir = os.path.join(root_dir, area, 'pano', 'rgb')
            mask_dir = os.path.join(root_dir, area, 'pano', self.mask_folder_name)
            
            if not os.path.exists(rgb_dir) or not os.path.exists(mask_dir): continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            for rgb_path in rgb_files:
                file_name = os.path.basename(rgb_path)
                mask_name = file_name.replace('_rgb.png', f'_{self.mask_folder_name}.png') 
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.filenames.append({'rgb': rgb_path, 'mask': mask_path})

        if debug_limit: self.filenames = self.filenames[:debug_limit]
        print(f"Found {len(self.filenames)} pairs.")

        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        v_centers = torch.linspace(90 - self.v_fov/2, -90 + self.v_fov/2, self.v_steps)
        u_centers = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1) 

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)

        # [新增] 定义光度增强 (Photometric Distortions)
        # 这些增强只改变像素值，不改变几何位置，所以 Mask 不需要跟着变
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.filenames[idx]
        try:
            img_pil = Image.open(data['rgb']).convert('RGB')
            mask_pil = Image.open(data['mask']).convert('RGB') # 读取 Mask

            # Resize
            if self.args and hasattr(self.args, 'pano_h'):
                target_h, target_w = self.args.pano_h, self.args.pano_w
                if img_pil.size != (target_w, target_h):
                    img_pil = img_pil.resize((target_w, target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)
            
            # =========================================================
            # [增强 1] 光度增强 (全局应用)
            # =========================================================
            if self.is_train:
                # 1. 颜色抖动 & 灰度 (概率应用)
                if random.random() < 0.8: # 80% 概率做颜色增强
                    img_pil = self.color_aug(img_pil)
                
                # 2. 高斯模糊 (模拟失焦)
                if random.random() < 0.1: # 10% 概率
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))

            img_np = np.array(img_pil)
            mask_raw = np.array(mask_pil)
            # 解析 Mask ID (假设 R通道即 ID)
            mask_np = mask_raw[:, :, 0].astype(np.int64)

            # =========================================================
            # [增强 2] 几何增强 (同时作用于 Img 和 Mask)
            # =========================================================
            if self.is_train:
                # 1. 水平滚动 (Horizontal Roll) - 核心全景增强
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1)
                mask_np = np.roll(mask_np, roll_idx, axis=1)
                
                # 2. 水平翻转 (Horizontal Flip) - 镜像
                # 注意：如果还需要输出法线，这里需要反转法线的 X 分量。纯分割任务无需担心。
                if random.random() < 0.5:
                    img_np = np.flip(img_np, axis=1)
                    mask_np = np.flip(mask_np, axis=1)

            # Extraction
            raw_h = img_np.shape[0] // self.v_steps
            raw_w = img_np.shape[1] // self.u_steps
            target_size = (int(raw_w), int(raw_h)) if raw_h != raw_w else int(raw_h)

            rgb_patches_list = odi.extract_patches_from_pano(
                pano_np=img_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            mask_patches_list = odi.extract_patches_from_pano(
                pano_np=mask_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            rgb_tensors = []
            mask_tensors = []
            
            for i in range(len(rgb_patches_list)):
                # Patch 已经是增强过的了，直接转 Tensor
                p_rgb = Image.fromarray(rgb_patches_list[i])
                p_rgb_tensor = self.normalize(transforms.ToTensor()(p_rgb))
                rgb_tensors.append(p_rgb_tensor)

                p_mask_np = mask_patches_list[i]
                mask_tensors.append(torch.from_numpy(p_mask_np.astype(np.int64)))

            patches_img = torch.stack(rgb_tensors)   
            patches_mask = torch.stack(mask_tensors) 

        except Exception as e:
            print(f"[Error] Loading {data.get('rgb', 'unknown')}: {e}")
            h_fb = self.args.pano_h // self.v_steps if self.args else 128
            w_fb = self.args.pano_w // self.u_steps if self.args else 128
            return (torch.zeros(self.u_steps * self.v_steps, 3, h_fb, w_fb),
                    self.angle_centers,
                    torch.zeros(self.u_steps * self.v_steps, h_fb, w_fb).long())

        return patches_img, self.angle_centers, patches_mask

def build_segmentation_dataset(is_train, args):
    return Stanford2D3DDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=None,
        is_train=is_train,
        num_classes=args.nb_classes,
        args=args
    )