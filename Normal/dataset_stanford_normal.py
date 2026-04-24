import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from torchvision import transforms
import torch.nn.functional as F
import random

# ==============================================================================
# 1. 移植 6通道代码中的核心 ODI 几何类 (确保几何逻辑一致)
# ==============================================================================

def rnd(x):
    if isinstance(x, (float, np.float32, np.float64)): return int(round(x))
    return (x+0.5).astype(int)

def limit_values(x, r):
    x = x.copy()
    x[x<r[0]] = r[0]
    x[x>r[1]] = r[1]
    return x

def polar(cord):
    if cord.ndim == 1: P = np.linalg.norm(cord)
    else: P = np.linalg.norm(cord, axis=0)
    phi = np.arcsin(cord[2] / (P + 1e-8))
    theta = np.arctan2(cord[1], cord[0])
    return [theta, phi]

class DatasetCameraPrm:
    def __init__(self, camera_angle, image_plane_size, view_angle):
        self.camera_angle = camera_angle
        self.image_plane_size = image_plane_size
        self.view_angle = view_angle
        L = (image_plane_size[0] / 2.0) / np.tan(view_angle[0] / 2.0)
        self.L = L
        ca = camera_angle
        
        self.nc = np.array([
            np.cos(ca[1]) * np.cos(ca[0]), 
            -np.cos(ca[1]) * np.sin(ca[0]), 
            np.sin(ca[1])
        ])
        
        self.xn = np.array([
            -np.sin(ca[0]), 
            -np.cos(ca[0]),
            0
        ])
        
        self.yn = np.array([
            -np.sin(ca[1]) * np.cos(ca[0]), 
            np.sin(ca[1]) * np.sin(ca[0]),
            np.cos(ca[1])
        ])
        
        self.c0 = self.L * self.nc

class DatasetOmniImage:
    def __init__(self, img_np): 
        self.img = img_np 
        
    def extract(self, prm):
        H, W = self.img.shape[:2]
        w, h = int(prm.image_plane_size[0]), int(prm.image_plane_size[1])
        
        c1, r1 = np.meshgrid(np.arange(w), np.arange(h))
        xp = c1 - (w - 1) / 2.0
        yp = (h - 1) / 2.0 - r1 
        
        p = xp[..., None] * prm.xn + yp[..., None] * prm.yn + prm.c0
        p_flat = p.reshape(-1, 3).T
        theta, phi = polar(p_flat)
        
        u = (theta / (2*np.pi) + 0.5) * W - 0.5
        v = (-phi / np.pi + 0.5) * H - 0.5
        
        u_int = limit_values(rnd(u), (0, W-1))
        v_int = limit_values(rnd(v), (0, H-1))
        
        out = self.img[v_int, u_int].reshape(h, w, -1)
        return out

# ==============================================================================
# 2. Dataset Implementation (适配 3通道 + 增强)
# ==============================================================================

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
        
        print(f"[{'Train' if is_train else 'Val'}] Stanford2D3D Normal (With Augmentation, No Flip)...")

        for area in areas:
            rgb_dir = os.path.join(root_dir, area, 'pano', 'rgb')
            norm_dir = os.path.join(root_dir, area, 'pano', self.target_folder_name)
            
            if not os.path.exists(rgb_dir) or not os.path.exists(norm_dir): continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            for rgb_path in rgb_files:
                f_name = os.path.basename(rgb_path)
                n_name = f_name.replace('_rgb.png', '_normals.png')
                n_path = os.path.join(norm_dir, n_name)
                
                if not os.path.exists(n_path):
                     n_path = os.path.join(norm_dir, f_name.replace('_rgb.png', '_normal.png'))
                
                if os.path.exists(n_path):
                    self.filenames.append({'rgb': rgb_path, 'normal': n_path})

        if debug_limit: self.filenames = self.filenames[:debug_limit]
        print(f"Found {len(self.filenames)} pairs.")

        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        
        # [新增] 光度增强定义 (仅改变 RGB 颜色/纹理，不影响几何结构)
        # 相比分割任务，法线对纹理细节更敏感，所以参数稍微保守一点点
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
        ])
        
        angle_list = []
        phis = np.linspace(np.pi/2 - np.radians(self.v_fov)/2, -np.pi/2 + np.radians(self.v_fov)/2, self.v_steps)
        thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)
        
        for phi_c in phis:
            for theta_c in thetas:
                angle_list.append([theta_c, phi_c]) # [Lon, Lat]
        
        self.angle_centers = torch.tensor(np.degrees(angle_list)).float()

    def __len__(self):
        return len(self.filenames)

    def get_cam_rotation_matrix(self, prm):
        fwd = prm.nc 
        right = prm.xn 
        up = -prm.yn 
        R = np.stack([right, up, fwd], axis=0)
        return R

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

            # =========================================================
            # [增强 1] 光度增强 (Photometric Augmentation)
            # =========================================================
            # 必须在转换为 Numpy 之前对 PIL Image 进行，确保全图一致性
            if self.is_train:
                # 1. 颜色抖动 & 灰度 (80% 概率)
                if random.random() < 0.8:
                    img_pil = self.color_aug(img_pil)
                
                # 2. 高斯模糊 (10% 概率，模拟失焦)
                if random.random() < 0.1:
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
            
            img_np = np.array(img_pil)
            
            norm_raw = np.array(norm_pil).astype(np.float32) / 255.0
            norm_swapped = norm_raw.copy()
            norm_swapped[..., 1] = norm_raw[..., 2] # New Y = Old Z
            norm_swapped[..., 2] = norm_raw[..., 1] # New Z = Old Y (Up)
            
            norm_world = norm_swapped * 2.0 - 1.0
            norm_world = norm_world / (np.linalg.norm(norm_world, axis=2, keepdims=True) + 1e-8)
            
            valid_mask = (img_np.sum(axis=2) > 10).astype(np.float32)

            # =========================================================
            # [增强 2] 几何增强 (Horizontal Roll)
            # =========================================================
            current_angles = self.angle_centers.clone() 
            if self.is_train and getattr(self.args, 'use_horizontal_roll', False):
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                
                # 图像和法线图同时滚动
                img_np = np.roll(img_np, roll_idx, axis=1)
                norm_world = np.roll(norm_world, roll_idx, axis=1)
                valid_mask = np.roll(valid_mask, roll_idx, axis=1)
                
                # 角度修正 (保持原有的修正逻辑)
                # 图像右移 -> 视口相对左移 -> 角度减小
                roll_deg = (roll_idx / img_np.shape[1]) * 360.0
                current_angles[:, 0] = current_angles[:, 0] - roll_deg
                current_angles[:, 0] = ((current_angles[:, 0] + 180) % 360) - 180
            
            # 注意：此处未添加 Horizontal Flip，以避免坐标系手性反转问题

            imc_rgb = DatasetOmniImage(img_np)
            imc_norm = DatasetOmniImage(norm_world)
            imc_mask = DatasetOmniImage(valid_mask)

            out_w = img_np.shape[1] // self.u_steps
            out_h = img_np.shape[0] // self.v_steps
            
            p_rgb_list, p_norm_list, p_mask_list = [], [], []

            phis = np.linspace(np.pi/2 - np.radians(self.v_fov)/2, -np.pi/2 + np.radians(self.v_fov)/2, self.v_steps)
            thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)

            for phi_c in phis:
                for theta_c in thetas:
                    prm = DatasetCameraPrm([theta_c, phi_c], [out_w, out_h], [np.radians(self.h_fov), np.radians(self.v_fov)])
                    
                    rgb_p = imc_rgb.extract(prm)
                    norm_p = imc_norm.extract(prm) 
                    mask_p = imc_mask.extract(prm)

                    R = self.get_cam_rotation_matrix(prm)
                    norm_c = np.dot(norm_p.reshape(-1, 3), R.T).reshape(out_h, out_w, 3)
                    
                    t_rgb = self.normalize(transforms.ToTensor()(Image.fromarray(rgb_p.astype(np.uint8))))
                    
                    t_norm = torch.from_numpy(norm_c).permute(2, 0, 1).float()
                    t_norm = F.normalize(t_norm, dim=0, p=2)
                    
                    t_mask = torch.from_numpy(mask_p).float()
                    if t_mask.ndim == 2: t_mask = t_mask.unsqueeze(0)
                    elif t_mask.shape[-1] == 1: t_mask = t_mask.permute(2, 0, 1)

                    p_rgb_list.append(t_rgb)
                    p_norm_list.append(t_norm)
                    p_mask_list.append(t_mask)

            return torch.stack(p_rgb_list), current_angles, torch.stack(p_norm_list), torch.stack(p_mask_list)

        except Exception as e:
            print(f"[Dataset Error] {e}")
            h_fb = self.args.pano_h // self.v_steps if self.args else 128
            w_fb = self.args.pano_w // self.u_steps if self.args else 128
            return torch.zeros(self.u_steps*self.v_steps, 3, h_fb, w_fb), self.angle_centers, \
                   torch.zeros(self.u_steps*self.v_steps, 3, h_fb, w_fb), torch.zeros(self.u_steps*self.v_steps, 1, h_fb, w_fb)

def build_normal_dataset(is_train, args):
    return Stanford2D3DDataset(root_dir=args.data_path if is_train else args.val_data_path,
                                 grid_height=args.grid_height, is_train=is_train, args=args)