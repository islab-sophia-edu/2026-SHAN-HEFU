import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import math
# 必须调用这个未修改的原始库
import util.odi_processing as odi 

class PanoramicDataset(Dataset):
    def __init__(self, root_dir, 
                 grid_height=4, 
                 pano_h=512, 
                 pano_w=1024, 
                 multiscale_sampling=True,    
                 use_horizontal_roll=True,    
                 is_training=True,
                 img_size=None,
                 device=None):
        
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.is_training = is_training
        
        self.pano_h = pano_h
        self.pano_w = pano_w
        self.grid_height = grid_height
        
        # 自动计算 Patch Size
        if img_size is None:
            self.patch_size = self.pano_h // self.grid_height
        else:
            self.patch_size = img_size
            
        # 自动获取设备 (GPU)
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"Dataset (Hybrid-ODI): Input {pano_w}x{pano_h}, Patch {self.patch_size}, Train: {is_training}, Device: {self.device}")

        # Grid Settings
        self.v_num = self.grid_height
        self.u_num = 2 * self.grid_height
        self.base_v_fov = 180.0 / self.v_num
        self.base_h_fov = 360.0 / self.u_num
        
        # 预计算 Grid Angles
        self.base_angles = self._generate_grid_angles()
        
        # 增强开关
        self.multiscale_sampling = multiscale_sampling and is_training
        self.use_horizontal_roll = use_horizontal_roll and is_training
        
        # ImageNet Norm (GPU Tensor)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _generate_grid_angles(self):
        # 这一步只在 init 做一次
        phis = np.linspace(np.pi/2 - np.radians(self.base_v_fov)/2, 
                           -np.pi/2 + np.radians(self.base_v_fov)/2, 
                           self.v_num)
        thetas = np.linspace(-np.pi, np.pi, self.u_num, endpoint=False)
        grid_phis, grid_thetas = np.meshgrid(phis, thetas, indexing='ij')
        
        # (N, 2)
        angles = np.stack([grid_thetas.flatten(), grid_phis.flatten()], axis=1)
        return angles # Numpy array

    def _get_random_params(self):
        theta_shift = 0.0
        scale = 1.0
        if self.is_training:
            if self.use_horizontal_roll:
                theta_shift = np.random.uniform(-np.pi, np.pi)
            if self.multiscale_sampling:
                scale = np.random.uniform(0.8, 1.2)
        return theta_shift, scale

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. IO (CPU)
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            pano_pil = Image.open(img_path).convert('RGB')
            # Resize
            if pano_pil.size != (self.pano_w, self.pano_h):
                pano_pil = pano_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)
            pano_np = np.array(pano_pil)
            
            # Roll (Augmentation)
            if self.use_horizontal_roll:
                shift = np.random.randint(0, self.pano_w)
                pano_np = np.roll(pano_np, shift, axis=1)
                
        except Exception as e:
            print(f"Error: {e}")
            pano_np = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.uint8)

        # 2. Upload to GPU (Early!)
        # (1, 3, H, W) normalized [0,1]
        pano_tensor = torch.from_numpy(pano_np).permute(2, 0, 1).float().to(self.device, non_blocking=True) / 255.0
        pano_tensor = pano_tensor.unsqueeze(0) 

        # 3. Augmentation Params
        theta_shift, scale = self._get_random_params()
        
        # 注意：这里计算 FOV 是为了传给 odi
        current_h_fov = np.radians(self.base_h_fov * scale)
        current_v_fov = np.radians(self.base_v_fov * scale)
        
        patches_list = []
        angles_list = []
        
        # 4. Loop Patches
        for i, angle in enumerate(self.base_angles):
            # Numpy Angles
            base_theta = angle[0]
            base_phi = angle[1]
            theta_c = base_theta + theta_shift
            
            # [关键点] 调用原始 ODI 计算几何 (CPU/Numpy)
            # prm.p 是 (3, patch_h, patch_w) 的世界坐标系点
            prm = odi.CameraPrm(
                camera_angle=[theta_c, base_phi], 
                view_angle=[current_h_fov, current_v_fov],
                image_plane_size=[self.patch_size, self.patch_size]
            )
            
            # 获取 3D 坐标并转到 GPU (这就是您说的"计算放过去")
            # prm.p 是 numpy array
            p_3d_np = prm.p 
            p_3d = torch.from_numpy(p_3d_np).float().to(self.device, non_blocking=True) # (3, H, W)

            # --- 以下是把 ODI 的 extract 逻辑搬到 Dataset 里用 GPU 实现 ---
            
            # A. 计算 XYZ Map (6通道输入之一)
            # 归一化 p 得到单位向量
            norm = torch.norm(p_3d, dim=0, keepdim=True)
            xyz_patch = p_3d / (norm + 1e-8) # (3, H, W)
            
            # B. 计算采样网格 (Sampling Grid)
            # 相当于 odi.polar() 的 GPU 实现
            x = xyz_patch[0]
            y = xyz_patch[1]
            z = xyz_patch[2]
            
            # Cartesian -> Spherical
            # 匹配 odi 的逻辑: phi = arcsin(z), theta = ...
            # 注意 odi 的 y 定义可能是反的，但 prm.p 已经是生成好的坐标，直接转即可
            # odi: theta = arccos(x / sqrt(x^2+y^2)) ...
            # 我们直接用 atan2 更稳
            lon = torch.atan2(-y, x) # [-pi, pi]
            lat = torch.asin(torch.clamp(z, -1, 1)) # [-pi/2, pi/2]
            
            # Spherical -> UV Grid [-1, 1]
            u = lon / math.pi
            v = -lat / (math.pi / 2.0)
            
            grid = torch.stack([u, v], dim=-1).unsqueeze(0) # (1, H, W, 2)
            
            # C. GPU 极速采样 RGB
            # 这一步替代了 odi.extract 里的 numpy indexing
            rgb_patch = F.grid_sample(pano_tensor, grid, mode='bilinear', align_corners=False).squeeze(0)
            
            # Normalize RGB
            rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
            
            # Concat RGB + XYZ
            full_patch = torch.cat([rgb_patch, xyz_patch], dim=0) # (6, H, W)
            patches_list.append(full_patch)
            
            # Record Angles
            deg_theta = np.degrees(theta_c)
            deg_phi = np.degrees(base_phi)
            deg_theta = (deg_theta + 180) % 360 - 180
            angles_list.append(torch.tensor([deg_theta, deg_phi], dtype=torch.float32))

        views = torch.stack(patches_list) # (N, 6, H, W)
        angles = torch.stack(angles_list) # (N, 2)
        
        # Weights
        weights = torch.cos(torch.deg2rad(angles[:, 1]))
        weights = weights / (weights.mean() + 1e-6)
        
        return views, angles, weights