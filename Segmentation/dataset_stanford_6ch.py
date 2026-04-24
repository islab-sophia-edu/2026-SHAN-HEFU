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
                 num_classes=14, debug_limit=None, args=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.num_classes = num_classes
        self.args = args 

        if is_train:
            areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
        else:
            areas = ['area_5a', 'area_5b']

        self.filenames = []
        self.mask_folder_name = 'anno' 
        
        print(f"[{'Train' if is_train else 'Val'}] Loading Stanford2D3D (6CH + Aggressive Aug)...")

        for area in areas:
            rgb_dir = os.path.join(root_dir, area, 'pano', 'rgb')
            mask_dir = os.path.join(root_dir, area, 'pano', self.mask_folder_name)
            
            if not os.path.exists(rgb_dir) or not os.path.exists(mask_dir):
                continue

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
        
        # Grid Center Calculation
        v_centers = torch.linspace(90 - self.v_fov/2, -90 + self.v_fov/2, self.v_steps)
        u_centers = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1) 

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        
        # Color Jitter (Stronger)
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.filenames[idx]
        
        try:
            img_pil = Image.open(data['rgb']).convert('RGB')
            mask_pil = Image.open(data['mask']).convert('RGB')

            # 1. Resize (加快 IO，如果是大图训练可以调大)
            if self.args and hasattr(self.args, 'pano_h'):
                target_h, target_w = self.args.pano_h, self.args.pano_w
                if img_pil.size != (target_w, target_h):
                    img_pil = img_pil.resize((target_w, target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)

            img_np = np.array(img_pil)
            mask_raw = np.array(mask_pil) 
            mask_np = mask_raw[:, :, 0].astype(np.int64)

            # ==========================================
            # 2. Aggressive Global Augmentation
            # ==========================================
            if self.is_train:
                # A. Horizontal Roll (0 ~ W)
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1)
                mask_np = np.roll(mask_np, roll_idx, axis=1)
                
                # B. Horizontal Flip (50%)
                if random.random() < 0.5:
                    img_np = np.fliplr(img_np)
                    mask_np = np.fliplr(mask_np)

            # ==========================================
            # 3. Aggressive Patch Augmentation parameters
            # ==========================================
            
            # [修改] 激进的 Scale (0.5 ~ 2.0)
            # 0.5 = Zoom In (看局部细节)
            # 2.0 = Zoom Out (看更大视野)
            if self.is_train:
                scale = random.uniform(0.5, 2.0)
            else:
                scale = 1.0
            
            # 计算当前 FOV
            h_fov_rad = np.radians(self.h_fov * scale)
            v_fov_rad = np.radians(self.v_fov * scale)

            # 4. 准备提取
            raw_h = img_np.shape[0] // self.v_steps
            raw_w = img_np.shape[1] // self.u_steps
            
            if isinstance(raw_h, int):
                target_size = (int(raw_w), int(raw_h)) if raw_h != raw_w else int(raw_h)
            else:
                target_size = 52
            
            if isinstance(target_size, tuple):
                out_w, out_h = target_size
            else:
                out_w, out_h = target_size, target_size

            imc_rgb = odi.OmniImage(img_np)
            imc_mask = odi.OmniImage(mask_np)

            patches_list = []
            masks_list = []

            # 5. 循环提取 (RGB + Mask + XYZ)
            phis = np.linspace(np.pi/2 - np.radians(self.v_fov)/2, 
                               -np.pi/2 + np.radians(self.v_fov)/2, 
                               self.v_steps)
            thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)
            
            for phi_c in phis:
                for theta_c in thetas:
                    
                    # [新增] Center Jitter (视角抖动)
                    # 防止模型死记硬背固定的 Grid 位置
                    # 抖动范围：Patch 自身大小的 +/- 20%
                    if self.is_train:
                        theta_jitter = random.uniform(-0.2, 0.2) * np.radians(self.h_fov)
                        phi_jitter = random.uniform(-0.2, 0.2) * np.radians(self.v_fov)
                        
                        current_theta = theta_c + theta_jitter
                        current_phi = np.clip(phi_c + phi_jitter, -np.pi/2 + 0.01, np.pi/2 - 0.01)
                    else:
                        current_theta = theta_c
                        current_phi = phi_c

                    prm = odi.CameraPrm(
                        camera_angle=[current_theta, current_phi],
                        view_angle=[h_fov_rad, v_fov_rad],
                        image_plane_size=[out_w, out_h]
                    )
                    
                    # A. Extract RGB
                    rgb_patch_np = imc_rgb.extract(prm)
                    
                    # B. Extract Mask
                    mask_patch_np = imc_mask.extract(prm)
                    if mask_patch_np.ndim == 3: mask_patch_np = mask_patch_np[:,:,0]

                    # C. Calculate XYZ
                    p_3d = prm.p # (3, H, W)
                    norm = np.linalg.norm(p_3d, axis=0, keepdims=True)
                    xyz_patch_np = p_3d / (norm + 1e-8) 
                    xyz_patch_np = xyz_patch_np.transpose(1, 2, 0) # (H, W, 3)

                    # D. To Tensor
                    p_rgb = Image.fromarray(rgb_patch_np)
                    
                    # [Stronger] Color Jitter (80% probability)
                    if self.is_train and random.random() < 0.8:
                        p_rgb = self.color_jitter(p_rgb)
                    
                    t_rgb = transforms.ToTensor()(p_rgb)
                    t_rgb = self.normalize(t_rgb)
                    
                    t_xyz = torch.from_numpy(xyz_patch_np).float().permute(2, 0, 1)
                    t_full = torch.cat([t_rgb, t_xyz], dim=0) # (6, H, W)
                    patches_list.append(t_full)
                    
                    t_mask = torch.from_numpy(mask_patch_np.astype(np.int64))
                    masks_list.append(t_mask)

            patches_img = torch.stack(patches_list)
            patches_mask = torch.stack(masks_list)

        except Exception as e:
            print(f"[Error] Loading {data.get('rgb', 'unknown')}: {e}")
            h_fb = 52
            w_fb = 52
            if self.args:
                h_fb = self.args.pano_h // self.v_steps
                w_fb = self.args.pano_w // self.u_steps
            return (torch.zeros(self.u_steps * self.v_steps, 6, h_fb, w_fb),
                    self.angle_centers,
                    torch.zeros(self.u_steps * self.v_steps, h_fb, w_fb).long())

        return patches_img, self.angle_centers, patches_mask

def build_segmentation_dataset(is_train, args):
    dataset = Stanford2D3DDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=None,
        is_train=is_train,
        num_classes=args.nb_classes,
        args=args
    )
    return dataset