import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import util.odi_processing as odi
import random

class PanoSegmentationDataset(Dataset):
    def __init__(self, root_dir, grid_height=16, img_size=None, is_train=True, 
                 num_classes=8, debug_limit=None, args=None):
        """
        root_dir: 数据集划分的根目录，例如 '/home/shanhefu/CVRG-Pano/train'
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.num_classes = num_classes
        self.args = args
        
        # [配置] 路径定义
        # 1. Mask 路径：直接使用划分后的目录 (train/anno 或 test/anno)
        self.mask_dir = os.path.join(root_dir, 'mask')
        
        # 2. RGB 路径：指向共享的大池子
        # 这里硬编码为你提供的路径，也可以改为从 args 传入
        self.rgb_shared_dir = '/home/shanhefu/CVRG-Pano/all-rgb'

        # [逻辑] 尺寸解析
        self.target_h = None
        self.target_w = None
        if self.args and hasattr(self.args, 'pano_h') and hasattr(self.args, 'pano_w'):
            self.target_h = self.args.pano_h
            self.target_w = self.args.pano_w
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.target_h, self.target_w = img_size

        print(f"[{'Train' if is_train else 'Val'}] Mask Source: {self.mask_dir}")
        print(f"[{'Train' if is_train else 'Val'}] RGB Source : {self.rgb_shared_dir}")

        # [核心修复] 防止数据泄露的关键步骤
        # 只读取 mask_dir 中的文件作为索引。
        # 只有在 mask_dir 里存在的 ID，才会被放入 Dataset。
        # 即使 all-rgb 里有 10000 张图，如果 train/anno 只有 500 张，这里也只加载 500 张。
        
        valid_mask_exts = {'.png'} # Mask 通常是 png
        self.file_pairs = [] # 存储 (mask_filename, rgb_filename) 的元组

        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Annotaion directory not found: {self.mask_dir}")

        # 获取所有 mask 文件名
        mask_files = sorted([
            f for f in os.listdir(self.mask_dir) 
            if os.path.splitext(f)[1].lower() in valid_mask_exts
        ])

        if debug_limit:
            mask_files = mask_files[:debug_limit]

        # [预处理] 建立 Mask -> RGB 的映射关系
        # 我们不能假设 RGB 也是 png，它可能是 jpg。需要扫描匹配。
        print(f"[{'Train' if is_train else 'Val'}] Matching RGB files for {len(mask_files)} masks...")
        
        missing_count = 0
        for mask_f in mask_files:
            basename = os.path.splitext(mask_f)[0]
            
            # 在 all-rgb 中尝试寻找对应的图片
            # 优先级: .jpg > .png > .jpeg
            found_rgb = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                candidate = basename + ext
                if os.path.exists(os.path.join(self.rgb_shared_dir, candidate)):
                    found_rgb = candidate
                    break
            
            if found_rgb:
                self.file_pairs.append((mask_f, found_rgb))
            else:
                missing_count += 1
                # 如果找不到 RGB，说明数据不完整，跳过或报错
                # print(f"Warning: RGB image for mask {mask_f} not found in {self.rgb_shared_dir}")

        print(f"[{'Train' if is_train else 'Val'}] Successfully paired {len(self.file_pairs)} images.")
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} masks have no corresponding RGB image in 'all-rgb'!")

        # 几何参数初始化 (保持不变)
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

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        # 获取匹配好的文件名对
        mask_name, rgb_name = self.file_pairs[idx]
        
        # 拼接完整路径
        mask_path = os.path.join(self.mask_dir, mask_name)
        img_path = os.path.join(self.rgb_shared_dir, rgb_name)
        
        try:
            # 读取
            img_pil = Image.open(img_path).convert('RGB')
            mask_pil = Image.open(mask_path)
            
            # [Resize 逻辑]
            if self.target_h is not None and self.target_w is not None:
                if img_pil.size != (self.target_w, self.target_h):
                    img_pil = img_pil.resize((self.target_w, self.target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((self.target_w, self.target_h), Image.NEAREST)
            
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)
            
            # 维度处理
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            
            # Data Augmentation (训练集专用)
            if self.is_train:
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1)
                mask_np = np.roll(mask_np, roll_idx, axis=1)

            # 动态计算 Patch 尺寸
            raw_h = img_np.shape[0] // self.v_steps
            raw_w = img_np.shape[1] // self.u_steps
            target_size = int(raw_h) if raw_h == raw_w else (int(raw_w), int(raw_h))

            # 提取 Patches
            rgb_patches_list = odi.extract_patches_from_pano(
                pano_np=img_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, 
                out_size=target_size
            )
            
            mask_patches_list = odi.extract_patches_from_pano(
                pano_np=mask_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, 
                out_size=target_size
            )
            
            rgb_tensors = []
            mask_tensors = []
            
            for i in range(len(rgb_patches_list)):
                p_rgb = Image.fromarray(rgb_patches_list[i])
                
                # Color Jitter
                if self.is_train and random.random() < 0.5:
                    p_rgb = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(p_rgb)
                
                p_rgb_tensor = transforms.ToTensor()(p_rgb)
                p_rgb_tensor = self.normalize(p_rgb_tensor)
                rgb_tensors.append(p_rgb_tensor)

                p_mask_np = mask_patches_list[i]
                if p_mask_np.ndim == 3: p_mask_np = p_mask_np[:,:,0]
                p_mask = torch.from_numpy(p_mask_np.astype(np.int64))
                mask_tensors.append(p_mask)

            patches_img = torch.stack(rgb_tensors)
            patches_mask = torch.stack(mask_tensors)

        except Exception as e:
            print(f"Error loading {mask_name} / {rgb_name}: {e}")
            # Fallback 逻辑
            h_fallback = self.target_h // self.v_steps if self.target_h else 52
            w_fallback = self.target_w // self.u_steps if self.target_w else 52
            return (torch.zeros(self.u_steps * self.v_steps, 3, h_fallback, w_fallback),
                    self.angle_centers,
                    torch.zeros(self.u_steps * self.v_steps, h_fallback, w_fallback).long())

        return patches_img, self.angle_centers, patches_mask

def build_segmentation_dataset(is_train, args):
    """
    args.data_path 应指向: /home/shanhefu/CVRG-Pano/train
    args.val_data_path 应指向: /home/shanhefu/CVRG-Pano/test
    """
    dataset = PanoSegmentationDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=args.input_size, 
        is_train=is_train,
        num_classes=args.nb_classes,
        args=args
    )
    return dataset