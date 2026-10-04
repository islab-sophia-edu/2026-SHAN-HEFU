import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

# rounding
def rnd(x):
    if isinstance(x, np.ndarray):
        return np.floor(x + 0.5).astype(int)
    return int(math.floor(float(x) + 0.5))
    
# polar cordinate: 
# cord.shepe=(3, c1, r1), cord[:,0,0]=[px, py, pz]
# or cord.shape = (3,), cord=[px, py, pz]
def polar(cord):
    if cord.ndim == 1:
        P = np.linalg.norm(cord)
    else:
        P = np.linalg.norm(cord, axis=0)
    z_over_p = np.clip(cord[2] / P, -1.0, 1.0)
    phi = np.arcsin(z_over_p)
    _eps = 1e-8
    denom = np.sqrt(cord[0]**2 + cord[1]**2 + _eps)
    theta_positive = np.arccos(np.clip(cord[0] / denom, -1.0, 1.0))
    theta_negative = -theta_positive
    theta = (cord[1] > 0) * theta_negative + (cord[1] <= 0) * theta_positive
    return [theta, phi]

# r = [lower, upper]
def limit_values(x, r, xcopy=1):
    if xcopy == 1:
        ret = x.copy()
    else:
        ret = x
    ret[ret<r[0]] = r[0]
    ret[ret>r[1]] = r[1]
    return ret


# calculating polar cordinates of image plane in omni-directional image using camera parameters
class CameraPrm:
    # camera_angle, view_angle: [horizontal, vertical]
    # L: distance from camera to image plane
    def __init__(self, camera_angle, image_plane_size=None, view_angle=None, L=None):
        # camera direction (in radians) [horizontal, vertical]
        self.camera_angle = camera_angle
        
        # view_angle: angle of view in radians [horizontal, vertical]
        # image_plane_size: [image width, image height]
        if view_angle is None:
            self.image_plane_size = image_plane_size
            self.L = L
            self.view_angle = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L))
        elif image_plane_size is  None:
            self.view_angle = view_angle
            self.L = L
            self.image_plane_size = 2.0 * L * np.tan(np.array(view_angle) / 2.0)
        else:
            self.image_plane_size = image_plane_size
            self.view_angle = view_angle
            L = (np.array(image_plane_size) / 2.0) / np.tan(np.array(view_angle) / 2.0)
            if rnd(L[0]) != rnd(L[1]):
                print('Warning: image_plane_size and view_angle are not matched.')
                va = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L[0]))
                ips = 2.0 * L[0] * np.tan(np.array(view_angle) / 2.0)
                print('image_plane_size should be (' + str(ips[0]) + ', ' + str(ips[1]) +
                      '), or view_angle should be (' +  str(math.degrees(va[0])) + ', ' + str(math.degrees(va[1])) + ').' )
                return
            else:
                self.L = L[0]
        
        # unit vector of cameara direction
        self.nc = np.array([
                np.cos(camera_angle[1]) * np.cos(camera_angle[0]), 
                -np.cos(camera_angle[1]) * np.sin(camera_angle[0]), 
                np.sin(camera_angle[1])
            ])
        
        # center of image plane
        self.c0 = self.L * self.nc

        # unit vector (xn, yn) in image plane
        self.xn = np.array([
                -np.sin(camera_angle[0]), 
                -np.cos(camera_angle[0]),
                0
            ])
        self.yn = np.array([
                -np.sin(camera_angle[1]) * np.cos(camera_angle[0]), 
                np.sin(camera_angle[1]) * np.sin(camera_angle[0]),
                np.cos(camera_angle[1])
            ])
        
        # meshgrid in image plane
        [c1, r1] = np.meshgrid(np.arange(0, rnd(self.image_plane_size[0])), np.arange(0, rnd(self.image_plane_size[1])))
       
        # 2d-cordinates in image [xp, yp]
        img_cord = [c1 - self.image_plane_size[0] / 2.0, -r1 + self.image_plane_size[1] / 2.0]
        
        # 3d-cordinatess in image plane [px, py, pz]
        self.p = self.get_3Dcordinate(img_cord)
        
        # polar cordinates in image plane [theta, phi]
        self.polar_omni_cord = polar(self.p)
        
    def get_3Dcordinate(self, c):
        [xp, yp] = c
        if type(xp) is np.ndarray: # xp, yp: array
            return xp * self.xn.reshape((3,1,1)) + yp * self.yn.reshape((3,1,1)) + np.ones(xp.shape) * self.c0.reshape((3,1,1))
        else: # xp, yp: scalars
            return xp * self.xn + yp * self.yn + self.c0

# omni-drectional image
class OmniImage:
    def __init__(self, omni_image):
        if type(omni_image) is str:
            self.omni_image = plt.imread(omni_image)
        else:
            self.omni_image = omni_image
    
    def extract(self, camera_prm):
        # 2d-cordinates in omni-directional image [c2, r2]
        c2 = (camera_prm.polar_omni_cord[0] / (2.0 * np.pi) + 1.0 / 2.0) * self.omni_image.shape[1] - 0.5
        r2 = (-camera_prm.polar_omni_cord[1] / np.pi + 1.0/2.0) * self.omni_image.shape[0] - 0.5
        #[c2_int, r2_int] = [rnd(c2), rnd(r2)]
        c2_int = limit_values(rnd(c2), (0, self.omni_image.shape[1]-1), 0)
        r2_int = limit_values(rnd(r2), (0, self.omni_image.shape[0]-1), 0)
        #self.omni_cord = [c2, r2]
        return self.omni_image[r2_int, c2_int]


def extract_patches_from_pano(pano_np, h_num, v_num, h_fov_deg, v_fov_deg, out_size):
    """ Extracts multiple perspective patches from a panoramic image. """
    patches = []
    imc = OmniImage(pano_np)
    
    # Generate non-redundant viewing angles
    thetas = np.linspace(-np.pi, np.pi, h_num, endpoint=False)
    v_fov_rad = np.radians(v_fov_deg)
    # 从大角度（顶部）到小角度（底部）
    start_phi = np.pi / 2.0 - v_fov_rad / 2.0
    end_phi = -np.pi / 2.0 + v_fov_rad / 2.0
    phis = np.linspace(start_phi, end_phi, v_num)

    for phi_c in phis:
        for theta_c in thetas:
            prm = CameraPrm(camera_angle=[theta_c, phi_c], view_angle=[np.radians(h_fov_deg), np.radians(v_fov_deg)], image_plane_size=[out_size, out_size])
            imp = imc.extract(prm)
            patches.append(imp)
            
    return patches

def embed_patches_to_pano(patches_np_list, pano_size, h_num, v_num, h_fov_deg, v_fov_deg):
    # 初始化全景画布
    pano_canvas = np.zeros((pano_size[0], pano_size[1], 3), dtype=np.float32)
    mask_all = np.zeros((pano_size[0], pano_size[1]), dtype=np.int32)
    
    # 生成相机角度
    thetas = np.linspace(-np.pi, np.pi, h_num, endpoint=False)
    v_fov_rad = np.radians(v_fov_deg)
    # 从大角度（顶部）到小角度（底部）
    start_phi = np.pi / 2.0 - v_fov_rad / 2.0
    end_phi = -np.pi / 2.0 + v_fov_rad / 2.0
    phis = np.linspace(start_phi, end_phi, v_num)
    
    # 预计算全景图的球面坐标
    c_omni, r_omni = np.meshgrid(np.arange(pano_size[1]), np.arange(pano_size[0]))
    theta_omni = (2.0 * (c_omni + 0.5) / float(pano_size[1]) - 1.0) * np.pi
    phi_omni = (0.5 - (r_omni + 0.5) / float(pano_size[0])) * np.pi
    
    # 全景图上每个点的单位方向向量
    pn = np.stack([
        np.cos(phi_omni) * np.cos(theta_omni),
        -np.cos(phi_omni) * np.sin(theta_omni),
        np.sin(phi_omni)
    ], axis=2)  # shape: (H, W, 3)
    
    patch_idx = 0
    for phi_c in phis:
        for theta_c in thetas:
            patch = patches_np_list[patch_idx]
            
            # 统一数据类型为float32 [0, 1]
            if patch.dtype == np.uint8:
                patch = patch.astype(np.float32) / 255.0
            elif patch.dtype != np.float32:
                patch = patch.astype(np.float32)
            
            h_patch, w_patch = patch.shape[:2]
            
            # 创建相机参数（与原始Embedding逻辑一致）
            prm = CameraPrm(
                camera_angle=[theta_c, phi_c],
                view_angle=[np.radians(h_fov_deg), np.radians(v_fov_deg)],
                image_plane_size=[w_patch, h_patch]
            )
            
            # 相机参数
            L = prm.L
            nc = prm.nc
            xn = prm.xn
            yn = prm.yn
            w1 = w_patch
            h1 = h_patch
            
            # 计算mask：哪些全景像素属于这个patch的视野
            cos_alpha = np.dot(pn, nc)  # shape: (H, W)
            mask = cos_alpha >= 2 * L / np.sqrt(w1**2 + h1**2 + 4*L**2)
            
            # 计算全景图像素到patch平面的映射
            r = np.zeros((pano_size[0], pano_size[1]))
            xp = np.zeros((pano_size[0], pano_size[1]))
            yp = np.zeros((pano_size[0], pano_size[1]))
            
            # 只计算mask内的点
            if np.any(mask):
                pn_masked = pn[mask]  # shape: (N, 3)
                r_masked = L / np.dot(pn_masked, nc)
                xp_masked = r_masked * np.dot(pn_masked, xn)
                yp_masked = r_masked * np.dot(pn_masked, yn)
                
                r[mask] = r_masked
                xp[mask] = xp_masked
                yp[mask] = yp_masked
            
            # 矩形mask：图像平面范围内
            mask = (mask) & (xp > -w1/2.0) & (xp < w1/2.0) & (yp > -h1/2.0) & (yp < h1/2.0)
            
            if not np.any(mask):
                patch_idx += 1
                continue
            
            # patch内的2D坐标（与原始embed一致）
            c1 = (w_patch / 2.0 + xp - 0.5) * mask
            r1 = (h_patch / 2.0 - yp - 0.5) * mask
            
            # 四舍五入并限制范围
            c1_int = limit_values(rnd(c1), (0, w_patch - 1), 0)
            r1_int = limit_values(rnd(r1), (0, h_patch - 1), 0)
            
            # 从patch采样并填充到全景图
            sampled_values = patch[r1_int, c1_int]  # shape: (H, W, 3)
            pano_canvas[mask] = sampled_values[mask]
            mask_all[mask] += 1
            
            patch_idx += 1
    
    # 检查覆盖情况
    overlapping = np.sum(mask_all > 1)
    missing = np.sum(mask_all == 0)
    total = pano_size[0] * pano_size[1]
    print(f'Overlapping dots: {overlapping}/{total}, Missing dots: {missing}/{total}')
    
    # 这里返回的是 [0, 1] 范围的float32图像
    return pano_canvas


import torch
import torch.nn.functional as F

def make_grid_gpu(tensor_list, grid_width, padding=4, bg_color=0):
    """
    GPU version of create_patch_grid_image.
    Input: list of tensors [C, H, W] or a stacked tensor [N, C, H, W]
    Output: Single tensor [C, Grid_H, Grid_W] in range [0, 1]
    """
    if isinstance(tensor_list, list):
        if len(tensor_list) == 0: return None
        patches = torch.stack(tensor_list)
    else:
        patches = tensor_list # Already [N, C, H, W]

    N, C, H, W = patches.shape
    grid_height = (N + grid_width - 1) // grid_width
    
    # Create canvas
    canvas_h = grid_height * (H + padding) + padding
    canvas_w = grid_width * (W + padding) + padding
    canvas = torch.full((C, canvas_h, canvas_w), bg_color, dtype=patches.dtype, device=patches.device)
    
    for i in range(N):
        row = i // grid_width
        col = i % grid_width
        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        canvas[:, y:y+H, x:x+W] = patches[i]
        
    return canvas


def embed_patches_to_pano_gpu(patches_tensor, pano_h, pano_w, h_num, v_num, h_fov_deg, v_fov_deg, angles):
    """
    GPU Accelerated version of embed_patches_to_pano.
    Logic is identical but executed on CUDA tensors.
    
    patches_tensor: [N, C, H, W], normalized or unnormalized (just copied)
    angles: [N, 2] (lon, lat) in degrees
    """
    device = patches_tensor.device
    N, C, H_patch, W_patch = patches_tensor.shape
    
    assert angles.shape[0] == N, f"Number of angles ({angles.shape[0]}) must match number of patches ({N})"
    
    assert H_patch == W_patch, "embed_patches_to_pano_gpu currently assumes square patches"
    assert abs(h_fov_deg - v_fov_deg) < 1e-3, "h_fov and v_fov must match for the current L derivation"
    
    # 1. Init Pano Canvas & Counter for Overlap Blending
    pano_canvas = torch.zeros((C, pano_h, pano_w), device=device, dtype=torch.float32)
    pano_count = torch.zeros((1, pano_h, pano_w), device=device, dtype=torch.float32)
    
    # 2. Pre-calculate Pano Spherical Coordinates (Vectorized)
    y_range = torch.arange(pano_h, device=device, dtype=torch.float32)
    x_range = torch.arange(pano_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    theta_omni = (2.0 * (grid_x + 0.5) / float(pano_w) - 1.0) * np.pi
    phi_omni = (0.5 - (grid_y + 0.5) / float(pano_h)) * np.pi        
    
    cos_phi = torch.cos(phi_omni)
    pn = torch.stack([
        cos_phi * torch.cos(theta_omni),
        -cos_phi * torch.sin(theta_omni), # Retained left-handed negative convention
        torch.sin(phi_omni)
    ], dim=0)

    # 3. Camera Parameters
    # Using python's math library instead of torch tensor to avoid unnecessary operations
    L = (W_patch / 2.0) / math.tan(math.radians(h_fov_deg / 2.0))
    
    for i in range(N):
        theta_c_deg = angles[i, 0]
        phi_c_deg = angles[i, 1]
        
        theta_rad = torch.deg2rad(theta_c_deg)
        phi_rad = torch.deg2rad(phi_c_deg)
        
        # [FIX 1] Replace list of 0-d tensors with torch.stack to clean autograd & avoid warnings
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
        cos_p, sin_p = torch.cos(phi_rad),   torch.sin(phi_rad)
        
        nc = torch.stack([
            cos_p * cos_t,
           -cos_p * sin_t,
            sin_p,
        ]).view(3, 1, 1)

        xn = torch.stack([
           -sin_t,
           -cos_t,
            torch.zeros_like(sin_t),
        ]).view(3, 1, 1)
        
        yn = torch.stack([
           -sin_p * cos_t,
            sin_p * sin_t,
            cos_p,
        ]).view(3, 1, 1)
        
        # 4. Compute intersection / mask
        cos_alpha = torch.sum(pn * nc, dim=0) 
        
        threshold = 2 * L / torch.sqrt(torch.tensor(W_patch**2 + H_patch**2 + 4*L**2, device=device))
        mask_fov = cos_alpha >= threshold
        
        if not mask_fov.any(): continue
        
        # 5. Project Pano pixels to Image Plane
        r_dist = L / (cos_alpha + 1e-8) 
        
        dot_xn = torch.sum(pn * xn, dim=0)
        dot_yn = torch.sum(pn * yn, dim=0)
        
        xp = r_dist * dot_xn
        yp = r_dist * dot_yn
        
        mask_rect = (xp > -W_patch/2.0) & (xp < W_patch/2.0) & \
                    (yp > -H_patch/2.0) & (yp < H_patch/2.0)
        
        final_mask = mask_fov & mask_rect
        
        if not final_mask.any(): continue
        
        # 6. Sample Colors
        def rnd_torch(x: torch.Tensor) -> torch.Tensor:
            return torch.floor(x + 0.5).long()
        
        c1 = torch.floor(W_patch / 2.0 + xp[final_mask]).long()
        r1 = torch.floor(H_patch / 2.0 - yp[final_mask]).long()

        c1 = torch.clamp(c1, 0, W_patch - 1)
        r1 = torch.clamp(r1, 0, H_patch - 1)
        
        values = patches_tensor[i, :, r1, c1] 
        
        # [FIX 3] Accumulate values for proper overlapping blend instead of overwrite
        pano_canvas[:, final_mask] += values
        pano_count[:, final_mask] += 1

    # [FIX 3] Divide accumulated colors by count for average blending in overlap areas
    pano_canvas = pano_canvas / pano_count.clamp(min=1.0)

    return pano_canvas