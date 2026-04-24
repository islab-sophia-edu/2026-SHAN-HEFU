import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# 引入 Dataset
from dataset_stanford_normal import build_normal_dataset

# 动态导入模型
try:
    from models_normal import vit_huge_patch14
except ImportError:
    try:
        from models_normal_cnn import vit_huge_patch14
    except ImportError:
        print("Error: Could not import model 'vit_huge_patch14'")
        sys.exit(1)

# ==============================================================================
# 1. GPU 几何核心：逆旋转 (Camera -> World)
# ==============================================================================

def inverse_rotate_batch_gpu(patches_cam, angles_deg):
    """
    将 Camera Space 的法线 Patch 批量逆旋转回 World Space。
    
    Args:
        patches_cam: (B, N, 3, H, W) or (N, 3, H, W)
        angles_deg:  (B, N, 2) or (N, 2) [theta, phi]
    Returns:
        patches_world: same shape as input
    """
    # 统一维度处理
    is_batch = patches_cam.ndim == 5
    if not is_batch:
        patches_cam = patches_cam.unsqueeze(0)
        angles_deg = angles_deg.unsqueeze(0)
    
    B, N, C, H, W = patches_cam.shape
    
    # 1. 准备旋转矩阵 (B, N, 3, 3)
    # 角度转弧度
    theta = torch.deg2rad(angles_deg[..., 0]) # (B, N)
    phi = torch.deg2rad(angles_deg[..., 1])   # (B, N)
    
    # 计算基向量 (与 Dataset 定义完全一致)
    # nc (Forward)
    nc_x = torch.cos(phi) * torch.cos(theta)
    nc_y = -torch.cos(phi) * torch.sin(theta)
    nc_z = torch.sin(phi)
    nc = torch.stack([nc_x, nc_y, nc_z], dim=-1) # (B, N, 3)
    
    # xn (Right) - Left-to-Right Scan (No Flip)
    xn_x = -torch.sin(theta)
    xn_y = -torch.cos(theta)
    xn_z = torch.zeros_like(theta)
    xn = torch.stack([xn_x, xn_y, xn_z], dim=-1) # (B, N, 3)
    
    # yn (Down/Up base)
    yn_x = -torch.sin(phi) * torch.cos(theta)
    yn_y = torch.sin(phi) * torch.sin(theta)
    yn_z = torch.cos(phi)
    yn = torch.stack([yn_x, yn_y, yn_z], dim=-1) # (B, N, 3)
    
    # 构造旋转矩阵 R_cam_to_world = R_world_to_cam.T
    # Dataset: R_w2c = [Right, Up, Forward] = [xn, -yn, nc]
    # R_c2w = R_w2c.T
    
    # Right=xn, Up=-yn, Fwd=nc
    row0 = xn   # (B, N, 3)
    row1 = -yn  # (B, N, 3)
    row2 = nc   # (B, N, 3)
    
    # R_w2c (3x3 matrix per patch)
    # stack dim=-2 -> (B, N, 3, 3)
    R_w2c = torch.stack([row0, row1, row2], dim=-2)
    
    # Inverse Rotation: v_world = R_w2c.T @ v_cam
    # Transpose last two dims
    R_c2w = R_w2c.transpose(-1, -2) # (B, N, 3, 3)
    
    # 2. 执行旋转
    # patches: (B, N, 3, H, W) -> (B, N, H, W, 3) -> (B, N, HW, 3, 1)
    p_flat = patches_cam.permute(0, 1, 3, 4, 2).reshape(B, N, H*W, 3).unsqueeze(-1)
    
    # 扩展矩阵以匹配像素数
    # R: (B, N, 1, 3, 3)
    R_exp = R_c2w.unsqueeze(2)
    
    # Matmul: (..., 3, 3) @ (..., 3, 1) -> (..., 3, 1)
    p_world_flat = torch.matmul(R_exp, p_flat).squeeze(-1) # (B, N, HW, 3)
    
    # Reshape back
    p_world = p_world_flat.view(B, N, H, W, 3).permute(0, 1, 4, 2, 3)
    
    # Normalize
    p_world = F.normalize(p_world, dim=2, p=2)
    
    if not is_batch:
        p_world = p_world.squeeze(0)
        
    return p_world

# ==============================================================================
# 2. GPU 拼接核心：embed_patches_to_pano_gpu
# ==============================================================================

def embed_patches_to_pano_gpu(patches_tensor, pano_h, pano_w, h_num, v_num, h_fov_deg, v_fov_deg, angles):
    """
    将 Patch 拼接到全景图 (GPU 版本)
    
    Args:
        patches_tensor: (N, C, H_p, W_p)
        angles: (N, 2) [theta, phi] in degrees
    """
    device = patches_tensor.device
    N, C, H_p, W_p = patches_tensor.shape
    
    # 1. 创建全景图画布坐标网格
    # Y: 0 ~ H-1, X: 0 ~ W-1
    grid_y, grid_x = torch.meshgrid(
        torch.arange(pano_h, device=device), 
        torch.arange(pano_w, device=device), 
        indexing='ij'
    )
    
    # 2. 全景图坐标 -> 球面坐标 (Phi, Theta)
    # Theta: -pi ~ pi (Left-Right)
    # Phi: pi/2 ~ -pi/2 (Top-Down)
    theta_omni = (2.0 * (grid_x.float() + 0.5) / pano_w - 1.0) * np.pi
    phi_omni = (0.5 - (grid_y.float() + 0.5) / pano_h) * np.pi
    
    # 3. 球面坐标 -> 3D 向量 (X, Y, Z)
    # 这一步生成全景图上每个像素对应的射线方向
    # System: Z-Up usually, but let's match Dataset logic
    # x = cos(phi)cos(theta), y = -cos(phi)sin(theta), z = sin(phi)
    # Wait, need to match Prm.nc definition
    # nc = [cos(phi)cos(theta), -cos(phi)sin(theta), sin(phi)]
    ray_x = torch.cos(phi_omni) * torch.cos(theta_omni)
    ray_y = -torch.cos(phi_omni) * torch.sin(theta_omni)
    ray_z = torch.sin(phi_omni)
    
    # (H_pano, W_pano, 3)
    rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
    
    # 4. 初始化输出
    canvas = torch.zeros((C, pano_h, pano_w), device=device, dtype=torch.float32)
    weight_map = torch.zeros((pano_h, pano_w), device=device, dtype=torch.float32) + 1e-6
    
    # 5. 逐 Patch 投影 (由于 Patch 数量不大，循环处理是可以的，或者分块处理)
    # 为了显存安全，我们还是逐个 Patch 或分批 Patch 投射
    # 但更高效的方法是：反向查找。
    # 这里为了简单复用 Dataset 逻辑，采用正向投射的思路有点难写 Shader。
    # 我们采用简化的“中心点距离”法或直接遍历 Patch。
    
    # 鉴于 N=128/512，循环 GPU 操作也很快
    
    # 预计算参数
    L = (W_p / 2.0) / np.tan(np.radians(v_fov_deg) / 2.0)
    
    # 角度转弧度
    theta_p = torch.deg2rad(angles[:, 0])
    phi_p = torch.deg2rad(angles[:, 1])
    
    # 计算每个 Patch 的相机基向量
    # nc (Forward)
    nc_x = torch.cos(phi_p) * torch.cos(theta_p)
    nc_y = -torch.cos(phi_p) * torch.sin(theta_p)
    nc_z = torch.sin(phi_p)
    nc_all = torch.stack([nc_x, nc_y, nc_z], dim=-1) # (N, 3)
    
    # xn (Right)
    xn_x = -torch.sin(theta_p)
    xn_y = -torch.cos(theta_p)
    xn_z = torch.zeros_like(theta_p)
    xn_all = torch.stack([xn_x, xn_y, xn_z], dim=-1) # (N, 3)
    
    # yn (Down/Up)
    yn_x = -torch.sin(phi_p) * torch.cos(theta_p)
    yn_y = torch.sin(phi_p) * torch.sin(theta_p)
    yn_z = torch.cos(phi_p)
    yn_all = torch.stack([yn_x, yn_y, yn_z], dim=-1) # (N, 3)
    
    # 核心循环
    for i in range(N):
        patch = patches_tensor[i] # (C, H, W)
        nc = nc_all[i]
        xn = xn_all[i]
        yn = yn_all[i]
        
        # 1. 粗筛选：找到全景图上与 Patch 中心方向接近的像素
        # 点积 > 阈值
        # threshold based on FOV
        min_cos = np.cos(np.radians(max(h_fov_deg, v_fov_deg) * 0.8)) # 稍微放宽一点
        
        # Dot product with all rays
        # rays: (H_pano, W_pano, 3), nc: (3)
        dot_prod = (rays * nc).sum(dim=-1)
        
        mask_rough = dot_prod > min_cos
        if not mask_rough.any(): continue
        
        # 2. 精确投影
        # Ray -> Image Plane
        # P_plane = R * P_ray (intersection)
        # r = L / (P_ray . nc)
        rays_valid = rays[mask_rough] # (K, 3)
        
        denom = (rays_valid * nc).sum(dim=-1)
        r = L / (denom + 1e-8)
        
        # Intersection point relative to center
        # P = r * Ray
        # proj_x = P . xn
        # proj_y = P . yn
        
        xp = r * (rays_valid * xn).sum(dim=-1)
        yp = r * (rays_valid * yn).sum(dim=-1)
        
        # Check boundary
        mask_valid = (xp > -W_p/2) & (xp < W_p/2) & (yp > -H_p/2) & (yp < H_p/2)
        
        if not mask_valid.any(): continue
        
        # Sample coordinates
        # c1 = (W_p / 2.0 + xp - 0.5)
        # r1 = (H_p / 2.0 - yp - 0.5)
        
        u_sample = (xp[mask_valid] + W_p/2 - 0.5) / (W_p - 1) * 2 - 1 # Normalize to [-1, 1] for grid_sample
        v_sample = (H_p/2 - yp[mask_valid] - 0.5) / (H_p - 1) * 2 - 1
        
        grid_sample = torch.stack([u_sample, v_sample], dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, K_valid, 2)
        
        # Grid Sample from Patch
        # patch: (C, H, W) -> (1, C, H, W)
        sampled_pixels = F.grid_sample(
            patch.unsqueeze(0), 
            grid_sample, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        ).squeeze(0).squeeze(1) # (C, K_valid)
        
        # Transpose to (K_valid, C)
        sampled_pixels = sampled_pixels.permute(1, 0)
        
        # Write back to canvas
        # Indices in pano
        indices_y, indices_x = torch.where(mask_rough)
        final_y = indices_y[mask_valid]
        final_x = indices_x[mask_valid]
        
        # Simple averaging for overlap
        # (Better: use distance weighting, but simple add is fast)
        # We need advanced indexing. Linear indices are faster.
        flat_indices = final_y * pano_w + final_x
        
        # 展平 canvas 以便写入
        canvas_flat = canvas.view(C, -1)
        weight_flat = weight_map.view(-1)
        
        # 累加颜色
        canvas_flat.index_add_(1, flat_indices, sampled_pixels.T)
        # 累加权重 (这里简单全1，也可以用 mask_valid 做软边缘)
        weight_flat.index_add_(0, flat_indices, torch.ones_like(final_y, dtype=torch.float32))
        
    # Normalize by weights
    canvas = canvas / weight_map.unsqueeze(0)
    
    # Fill missing with grey or black
    # mask_missing = weight_map < 1.0
    
    return canvas

# ==============================================================================
# Main Pipeline
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--pano_h', default=2048, type=int)
    parser.add_argument('--pano_w', default=4096, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', default='vit_huge_patch14', type=str)
    parser.add_argument('--resume', default='', type=str, required=True)
    parser.add_argument('--val_data_path', default='./test', type=str)
    parser.add_argument('--output_dir', default='./vis_output_gpu', type=str)
    parser.add_argument('--num_samples', default=5, type=int)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Geometry Params
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    h_fov = 360.0 / u_steps
    v_fov = 180.0 / v_steps
    img_size = args.pano_h // args.grid_height

    # 1. Dataset
    print(f"Dataset Path: {args.val_data_path}")
    dataset = build_normal_dataset(is_train=False, args=args)
    # Shuffle=False to compare specific samples easily
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2. Model
    print(f"Loading Model: {args.model}")
    # 动态加载模型，确保 in_chans=3
    model = globals()[args.model](
        img_size=img_size, patch_size=img_size, in_chans=3, 
        grid_height=args.grid_height, output_size=(args.pano_h, args.pano_w)
    )
    
    # 加载权重
    print(f"Loading Checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    # 去除前缀
    new_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    # 加载
    msg = model.load_state_dict(new_dict, strict=False)
    print(f"Model Loaded. Missing keys: {len(msg.missing_keys)}")
    
    model.to(device).eval()

    print(f"\n>>> Start Visualization (Total: {min(len(dataset), args.num_samples)} samples) <<<\n")

    for i, batch in enumerate(data_loader):
        if i >= args.num_samples: break
        
        # Unpack (B=1)
        views, angles, targets, masks = batch
        # views:   (1, N, 3, H, W) - RGB
        # angles:  (1, N, 2)       - Degrees
        # targets: (1, N, 3, H, W) - GT Normal (Camera Space)
        
        views = views.to(device)
        angles = angles.to(device)
        targets = targets.to(device)
        
        # A. Inference
        print(f"[Sample {i}] Inference...")
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            preds_cam = model(views, angles) # (1, N, 3, H, W)
        preds_cam = preds_cam.float()
        
        # B. Inverse Rotation (Camera -> World) [GPU]
        print(f"  > Inverse Rotation (GPU)...")
        # angles: (1, N, 2)
        preds_world = inverse_rotate_batch_gpu(preds_cam, angles) # (1, N, 3, H, W)
        targets_world = inverse_rotate_batch_gpu(targets, angles) # (1, N, 3, H, W)
        
        # C. Stitching [GPU]
        print(f"  > Stitching Panoramas (GPU)...")
        
        # 准备 RGB (反归一化)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        rgb_denorm = torch.clamp(views * std + mean, 0, 1) # (1, N, 3, H, W)
        
        # 取第一个 Batch
        # angles[0]: (N, 2)
        pano_rgb = embed_patches_to_pano_gpu(
            rgb_denorm[0], args.pano_h, args.pano_w, 
            u_steps, v_steps, h_fov, v_fov, angles[0]
        )
        
        pano_pred = embed_patches_to_pano_gpu(
            preds_world[0], args.pano_h, args.pano_w, 
            u_steps, v_steps, h_fov, v_fov, angles[0]
        )
        
        pano_gt = embed_patches_to_pano_gpu(
            targets_world[0], args.pano_h, args.pano_w, 
            u_steps, v_steps, h_fov, v_fov, angles[0]
        )
        
        # D. Visualization & Save
        print(f"  > Saving Image...")
        
        # Helper: Tensor (C, H, W) -> Numpy (H, W, C) & Norm Mapping
        def to_img(t, is_normal=False):
            arr = t.detach().cpu().permute(1, 2, 0).numpy()
            if is_normal:
                # [-1, 1] -> [0, 1]
                arr = np.clip((arr + 1.0) / 2.0, 0, 1)
            else:
                arr = np.clip(arr, 0, 1)
            return (arr * 255).astype(np.uint8)

        img_rgb = Image.fromarray(to_img(pano_rgb, is_normal=False))
        img_pred = Image.fromarray(to_img(pano_pred, is_normal=True))
        img_gt = Image.fromarray(to_img(pano_gt, is_normal=True))
        
        # Combine Vertically
        w, h = img_rgb.size
        combined = Image.new('RGB', (w, h * 3))
        combined.paste(img_rgb, (0, 0))
        combined.paste(img_pred, (0, h))
        combined.paste(img_gt, (0, h * 2))
        
        # Draw Labels (Optional, skip for pure visual)
        # ...
        
        save_path = os.path.join(args.output_dir, f"sample_{i:03d}_gpu.png")
        combined.save(save_path)
        print(f"✅ Saved to {save_path}")

    print("\nDone.")

if __name__ == '__main__':
    main()