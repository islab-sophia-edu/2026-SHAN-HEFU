import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import torch.nn.functional as F

# ==============================================================================
# ODI 核心逻辑 - 修复镜像扫描问题
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
        
        # [FIX] 移除 -1.0 翻转。使用标准 Left-to-Right 扫描，修复图像破碎问题。
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
        # [FIX] 使用标准中心对齐坐标
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
# DATASET
# ==============================================================================

class StanfordNormalDataset(Dataset):
    def __init__(self, root_dir, grid_height=16, img_size=None, is_train=True, 
                 debug_limit=None, args=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.args = args 

        areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'] if is_train else ['area_5a', 'area_5b']

        self.filenames = []
        self.target_folder_name = 'normal' 
        
        print(f"[{'Train' if is_train else 'Val'}] Stanford2D3D (Base Fixed: Y-Up Aligned)...")

        for area in areas:
            rgb_dir = os.path.join(root_dir, area, 'pano', 'rgb')
            norm_dir = os.path.join(root_dir, area, 'pano', self.target_folder_name)
            if not os.path.exists(rgb_dir) or not os.path.exists(norm_dir): continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            for rgb_path in rgb_files:
                f_name = os.path.basename(rgb_path)
                n_name = f_name.replace('_rgb.png', '_normal.png')
                n_path = os.path.join(norm_dir, n_name)
                if not os.path.exists(n_path):
                     n_path = os.path.join(norm_dir, f_name.replace('_rgb.png', '_normals.png'))
                if os.path.exists(n_path):
                    self.filenames.append({'rgb': rgb_path, 'normal': n_path})

        if debug_limit: self.filenames = self.filenames[:debug_limit]
        print(f"Found {len(self.filenames)} pairs.")

        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        angle_list = []
        phis = np.linspace(np.pi/2 - np.radians(self.v_fov)/2, -np.pi/2 + np.radians(self.v_fov)/2, self.v_steps)
        thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)
        for phi_c in phis:
            for theta_c in thetas:
                angle_list.append([theta_c, phi_c])
        self.angle_centers = torch.tensor(np.degrees(angle_list)).float()

    def __len__(self):
        return len(self.filenames)

    def get_cam_rotation_matrix_candidate_4(self, prm):
        # [FIX] 重新对齐坐标系映射
        # 目标：匹配 Y-Up 模型定义。
        # 标准映射：Right=xn, Up=-yn (yn原本向下), Forward=nc
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

            img_np = np.array(img_pil)
            # 原始 Normal 处理
            norm_world = (np.array(norm_pil).astype(np.float32) / 255.0) * 2.0 - 1.0
            # [FIX] 根据诊断报告确认垂直轴为 C1 (Y-Up)，保持原始手性
            norm_world = norm_world / (np.linalg.norm(norm_world, axis=2, keepdims=True) + 1e-8)
            
            valid_mask = (img_np.sum(axis=2) > 10).astype(np.float32)

            out_w, out_h = img_np.shape[1] // self.u_steps, img_np.shape[0] // self.v_steps
            imc_rgb, imc_norm, imc_mask = DatasetOmniImage(img_np), DatasetOmniImage(norm_world), DatasetOmniImage(valid_mask)

            # [STATIC XYZ] 修正为左向右，对齐像素
            c1, r1 = np.meshgrid(np.arange(out_w), np.arange(out_h))
            xp_s, yp_s = c1 - (out_w - 1) / 2.0, (out_h - 1) / 2.0 - r1
            L_s = (out_w / 2.0) / np.tan(np.radians(self.h_fov) / 2.0)
            ray = np.stack([xp_s, yp_s, np.full_like(xp_s, L_s)], axis=2)
            t_xyz_static = torch.from_numpy(ray / (np.linalg.norm(ray, axis=2, keepdims=True) + 1e-8)).float().permute(2, 0, 1)

            p_list, n_list, m_list = [], [], []
            phis = np.linspace(np.pi/2 - np.radians(self.v_fov)/2, -np.pi/2 + np.radians(self.v_fov)/2, self.v_steps)
            thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)

            for phi_c in phis:
                for theta_c in thetas:
                    prm = DatasetCameraPrm([theta_c, phi_c], [out_w, out_h], [np.radians(self.h_fov), np.radians(self.v_fov)])
                    
                    rgb_p = imc_rgb.extract(prm)
                    norm_p = imc_norm.extract(prm)
                    mask_p = imc_mask.extract(prm)

                    # 旋转进入相机局部空间
                    R = self.get_cam_rotation_matrix_candidate_4(prm)
                    norm_c = np.dot(norm_p.reshape(-1, 3), R.T).reshape(out_h, out_w, 3)

                    # 返回 Tensor
                    t_rgb = self.normalize(transforms.ToTensor()(Image.fromarray(rgb_p.astype(np.uint8))))
                    p_list.append(torch.cat([t_rgb, t_xyz_static], dim=0))
                    n_list.append(F.normalize(torch.from_numpy(norm_c).permute(2, 0, 1).float(), dim=0, p=2))
                    
                    # 确保 Mask 形状为 (1, H, W)
                    m_t = torch.from_numpy(mask_p).float()
                    if m_t.ndim == 2: m_t = m_t.unsqueeze(0)
                    elif m_t.shape[-1] == 1: m_t = m_t.permute(2, 0, 1)
                    m_list.append(m_t)

            return torch.stack(p_list), self.angle_centers, torch.stack(n_list), torch.stack(m_list)

        except Exception as e:
            print(f"[Dataset Error] {e}")
            # 回退处理，防止 DataLoader 崩溃
            h_fb = self.args.pano_h // self.v_steps if self.args else 128
            w_fb = self.args.pano_w // self.u_steps if self.args else 128
            return torch.zeros(self.u_steps*self.v_steps, 6, h_fb, w_fb), self.angle_centers, \
                   torch.zeros(self.u_steps*self.v_steps, 3, h_fb, w_fb), torch.zeros(self.u_steps*self.v_steps, 1, h_fb, w_fb)

def build_normal_dataset(is_train, args):
    return StanfordNormalDataset(root_dir=args.data_path if is_train else args.val_data_path,
                                 grid_height=args.grid_height, is_train=is_train, args=args)