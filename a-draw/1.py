import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------
# 1. 简化的核心函数 (提取自你提供的代码，确保独立运行)
# ---------------------------------------------------------
class CameraPrm:
    def __init__(self, camera_angle, view_angle, image_plane_size, L):
        self.nc = np.array([np.cos(camera_angle[1]) * np.cos(camera_angle[0]), -np.cos(camera_angle[1]) * np.sin(camera_angle[0]), np.sin(camera_angle[1])])
        self.xn = np.array([-np.sin(camera_angle[0]), -np.cos(camera_angle[0]), 0])
        self.yn = np.array([-np.sin(camera_angle[1]) * np.cos(camera_angle[0]), np.sin(camera_angle[1]) * np.sin(camera_angle[0]), np.cos(camera_angle[1])])
        self.L = L
        self.image_plane_size = image_plane_size
        self.c0 = self.L * self.nc

def extract_patches_simple(pano_img, h_num, v_num, h_fov_deg, v_fov_deg, out_size):
    # 简化版提取，核心逻辑一致
    H, W, _ = pano_img.shape
    patches = []
    
    # 模拟视场角对应的焦距 L
    view_angle = [np.radians(h_fov_deg), np.radians(v_fov_deg)]
    L = (out_size / 2.0) / np.tan(view_angle[0] / 2.0)
    
    # 生成网格坐标 (模拟 embed_patches_to_pano 的逻辑)
    # 注意：为了绘图好看，我们按 Grid 顺序输出
    patch_grid = np.zeros((v_num, h_num, out_size, out_size, 3), dtype=np.uint8)
    
    # 简单模拟切片过程 (真实提取需要上面的 CameraPrm 完整计算，
    # 这里为了演示绘图，我们直接在全景图上均匀采样，保证代码可运行且不需要庞大的依赖)
    step_x = W // h_num
    step_y = H // v_num
    
    for r in range(v_num):
        for c in range(h_num):
            # 简单裁切作为示意 (你可用你真实的 extract_patches 替换这里)
            y0, x0 = r * step_y, c * step_x
            crop = pano_img[y0:y0+step_y, x0:x0+step_x]
            crop_resized = np.array(Image.fromarray(crop).resize((out_size, out_size)))
            patch_grid[r, c] = crop_resized
            
    return patch_grid

# ---------------------------------------------------------
# 2. 执行数据生成
# ---------------------------------------------------------
def generate_viz_data():
    # A. 加载或生成一张全景图
    try:
        # 尝试加载本地图片，如果没有则生成彩虹图
        img = np.array(Image.open('input.jpg').convert('RGB')) 
        print("Loaded input.jpg")
    except:
        print("Generating synthetic panorama...")
        H, W = 512, 1024
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        X, Y = np.meshgrid(x, y)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[..., 0] = (np.sin(X * 10) * 127 + 128).astype(np.uint8)
        img[..., 1] = (np.cos(Y * 5) * 127 + 128).astype(np.uint8)
        img[..., 2] = ((X+Y) * 60 % 255).astype(np.uint8)

    # B. 参数设置 (对应你的 Vit_base_patch16)
    grid_h = 4  # v_num
    grid_w = 8  # h_num
    N = grid_h * grid_w
    patch_size = 64
    mask_ratio = 0.75
    
    # C. 提取 Patch
    # patch_grid shape: [grid_h, grid_w, H, W, 3]
    patch_grid = extract_patches_simple(img, grid_w, grid_h, 360/grid_w, 180/grid_h, patch_size)
    
    # D. 生成 Mask (模拟 Adaptive Masking 或 Random Masking)
    # 0 = Visible, 1 = Masked
    num_masked = int(N * mask_ratio)
    
    # 创建一个随机mask
    mask_flat = np.array([0] * (N - num_masked) + [1] * num_masked)
    np.random.shuffle(mask_flat)
    mask_grid = mask_flat.reshape(grid_h, grid_w)
    
    # E. 准备 MATLAB 数据
    # MATLAB 读取时 shape 为 (H, W, C, N_grid_row, N_grid_col)
    # 我们将其转置以适应 MATLAB 的习惯
    mat_patches = patch_grid.transpose(2, 3, 4, 0, 1) # [H, W, 3, Rows, Cols]
    
    data = {
        'patches': mat_patches,       # 原始图像切片
        'mask_layout': mask_grid,     # 掩码矩阵 (0/1)
        'patch_size': patch_size,
        'grid_rows': grid_h,
        'grid_cols': grid_w
    }
    
    sio.savemat('pano_mae_viz_data.mat', data)
    print("Data saved to pano_mae_viz_data.mat. Now run the MATLAB script.")

if __name__ == "__main__":
    generate_viz_data()