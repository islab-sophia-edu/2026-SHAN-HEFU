import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import util.odi_processing as odi
import random
import math

class PanoSegmentationDataset(Dataset):
    def __init__(self, root_dir, grid_height=16, img_size=None, is_train=True, 
                 num_classes=8, debug_limit=None, args=None):
        """
        CVRG-Pano Dataset with Aggressive 6-Channel Augmentation
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.num_classes = num_classes
        self.args = args
        
        # Paths
        self.mask_dir = os.path.join(root_dir, 'anno')
        self.rgb_shared_dir = '/home/shanhefu/CVRG-Pano/all-rgb'

        # Target Size
        self.target_h = None
        self.target_w = None
        if self.args and hasattr(self.args, 'pano_h') and hasattr(self.args, 'pano_w'):
            self.target_h = self.args.pano_h
            self.target_w = self.args.pano_w
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.target_h, self.target_w = img_size

        print(f"[{'Train' if is_train else 'Val'}] Mask Source: {self.mask_dir}")
        print(f"[{'Train' if is_train else 'Val'}] Mode: 6CH + Aggressive Augmentation")

        # File Matching
        valid_mask_exts = {'.png'} 
        self.file_pairs = [] 

        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Annotaion directory not found: {self.mask_dir}")

        mask_files = sorted([
            f for f in os.listdir(self.mask_dir) 
            if os.path.splitext(f)[1].lower() in valid_mask_exts
        ])

        if debug_limit:
            mask_files = mask_files[:debug_limit]

        print(f"Matching RGB files for {len(mask_files)} masks...")
        missing_count = 0
        for mask_f in mask_files:
            basename = os.path.splitext(mask_f)[0]
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

        print(f"Successfully paired {len(self.file_pairs)} images.")

        # Grid Params
        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        # Grid Centers
        v_centers = torch.linspace(90 - self.v_fov/2, -90 + self.v_fov/2, self.v_steps)
        u_centers = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1) 

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        
        # [新增] Stronger Color Jitter
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        mask_name, rgb_name = self.file_pairs[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        img_path = os.path.join(self.rgb_shared_dir, rgb_name)
        
        try:
            # 1. Load & Resize
            img_pil = Image.open(img_path).convert('RGB')
            mask_pil = Image.open(mask_path)
            
            if self.target_h is not None and self.target_w is not None:
                if img_pil.size != (self.target_w, self.target_h):
                    img_pil = img_pil.resize((self.target_w, self.target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((self.target_w, self.target_h), Image.NEAREST)
            
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)
            
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            
            # ==========================================
            # 2. Global Augmentation (Roll & Flip)
            # ==========================================
            if self.is_train:
                # A. Horizontal Roll
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1)
                mask_np = np.roll(mask_np, roll_idx, axis=1)
                
                # B. [新增] Horizontal Flip
                if random.random() < 0.5:
                    img_np = np.fliplr(img_np)
                    mask_np = np.fliplr(mask_np)

            # ==========================================
            # 3. Patch Scale Augmentation (Zoom In/Out)
            # ==========================================
            # 动态计算 Patch 尺寸
            raw_h = img_np.shape[0] // self.v_steps
            raw_w = img_np.shape[1] // self.u_steps
            target_size = int(raw_h) if raw_h == raw_w else (int(raw_w), int(raw_h))
            
            if isinstance(target_size, tuple):
                out_w, out_h = target_size
            else:
                out_w, out_h = target_size, target_size

            # [新增] Scale Jitter (0.5 ~ 2.0)
            if self.is_train:
                scale = random.uniform(0.5, 2.0)
            else:
                scale = 1.0
            
            h_fov_rad = np.radians(self.h_fov * scale)
            v_fov_rad = np.radians(self.v_fov * scale)

            # 4. Extraction Prep
            imc_rgb = odi.OmniImage(img_np)
            imc_mask = odi.OmniImage(mask_np)
            
            patches_list = []
            masks_list = []

            phis = np.linspace(np.pi/2 - np.radians(self.v_fov)/2, 
                               -np.pi/2 + np.radians(self.v_fov)/2, 
                               self.v_steps)
            thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)

            # 5. Loop & Extract (With Center Jitter)
            for phi_c in phis:
                for theta_c in thetas:
                    
                    # [新增] Center Jitter (视角微调，仅训练)
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
                    
                    # A. RGB
                    rgb_patch_np = imc_rgb.extract(prm)
                    
                    # B. Mask
                    mask_patch_np = imc_mask.extract(prm)
                    if mask_patch_np.ndim == 3: mask_patch_np = mask_patch_np[:,:,0]

                    # C. XYZ (6CH Key Feature)
                    p_3d = prm.p # (3, H, W)
                    norm = np.linalg.norm(p_3d, axis=0, keepdims=True)
                    xyz_patch_np = p_3d / (norm + 1e-8)
                    xyz_patch_np = xyz_patch_np.transpose(1, 2, 0) # (H, W, 3)

                    # D. Tensor & Augmentation
                    p_rgb = Image.fromarray(rgb_patch_np)
                    if self.is_train and random.random() < 0.8:
                        p_rgb = self.color_jitter(p_rgb)
                    
                    t_rgb = transforms.ToTensor()(p_rgb)
                    t_rgb = self.normalize(t_rgb)
                    
                    t_xyz = torch.from_numpy(xyz_patch_np).float().permute(2, 0, 1)
                    t_full = torch.cat([t_rgb, t_xyz], dim=0) # (6, H, W)
                    patches_list.append(t_full)
                    
                    t_mask = torch.from_numpy(mask_patch_np.astype(np.int64))
                    masks_list.append(t_mask)

            patches_img = torch.stack(patches_list) # (N, 6, H, W)
            patches_mask = torch.stack(masks_list)  # (N, H, W)

        except Exception as e:
            print(f"Error loading {mask_name} / {rgb_name}: {e}")
            h_fallback = self.target_h // self.v_steps if self.target_h else 52
            w_fallback = self.target_w // self.u_steps if self.target_w else 52
            return (torch.zeros(self.u_steps * self.v_steps, 6, h_fallback, w_fallback),
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