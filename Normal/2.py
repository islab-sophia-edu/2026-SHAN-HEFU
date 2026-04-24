import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 引入 Dataset
from dataset_stanford_normal import build_normal_dataset

# 动态导入模型
try:
    from models_normal_cnn import vit_huge_patch14 
except ImportError:
    from models_normal import vit_huge_patch14

# ==============================================================================
# 1. GPU 核心：ODI 几何拼接 + 逆旋转 (Camera -> World)
# ==============================================================================

@torch.no_grad()
def diagnostic_stitch_with_rotation_gpu(patches_tensor, angles_deg, pano_h, pano_w, h_fov_deg, v_fov_deg):
    """
    【最终修正版】
    核心修复：保持所有微分几何定义(nc/xn/yn)和旋转矩阵(R)不变，仅修正全景图的 theta 扫描方向。
    这解决了镜像问题，同时完美保留了法线向量的正确性。
    """
    device = patches_tensor.device
    N, C, H_p, W_p = patches_tensor.shape
    
    # 1. 预计算全景图网格
    y_range = torch.linspace(0, pano_h - 1, pano_h, device=device)
    x_range = torch.linspace(0, pano_w - 1, pano_w, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # --- [关键修改] 修正镜像问题 ---
    # 原逻辑: (2.0 * x / W - 1.0) -> Range [-1, 1] -> [-pi, pi]
    # 新逻辑: 负号反转 -> Range [1, -1] -> [pi, -pi]
    # 解释: 改变扫描方向，相当于水平翻转了全景图的"查找表"，从而抵消镜像，
    # 但不改变后续 pn/nc/xn/yn 之间的任何数学导数关系。
    theta_omni = - (2.0 * grid_x / (pano_w - 1) - 1.0) * torch.pi
    
    # phi 保持不变 (从上到下)
    phi_omni = (0.5 - grid_y / (pano_h - 1)) * torch.pi
    
    # 构造全景图射线 (公式保持严格不变，以维持与 nc/xn/yn 的一致性)
    cos_phi = torch.cos(phi_omni)
    pn = torch.stack([
        cos_phi * torch.cos(theta_omni),
        -cos_phi * torch.sin(theta_omni), # 保持原有的负号定义
        torch.sin(phi_omni)
    ], dim=0) 
    
    out_pano = torch.zeros((C, pano_h, pano_w), device=device)
    weight_map = torch.zeros((1, pano_h, pano_w), device=device)

    # 焦距计算
    L = (W_p / 2.0) / torch.tan(torch.deg2rad(torch.tensor(h_fov_deg / 2.0, device=device)))

    for i in range(N):
        theta_c = torch.deg2rad(angles_deg[i, 0])
        phi_c = torch.deg2rad(angles_deg[i, 1])
        
        # --- 保持完全不变的基向量定义 ---
        # 这里的数学关系是自洽的，改动任何符号都会破坏法线颜色
        nc = torch.tensor([
            torch.cos(phi_c) * torch.cos(theta_c),
            -torch.cos(phi_c) * torch.sin(theta_c),
            torch.sin(phi_c)
        ], device=device)
        
        xn = torch.tensor([-torch.sin(theta_c), -torch.cos(theta_c), 0.0], device=device)
        
        yn = torch.tensor([
            -torch.sin(phi_c) * torch.cos(theta_c),
            torch.sin(phi_c) * torch.sin(theta_c),
            torch.cos(phi_c)
        ], device=device)

        # 1. 筛选视野
        dot_nc = torch.sum(pn * nc.view(3, 1, 1), dim=0)
        mask_front = dot_nc > 0.1 
        if not mask_front.any(): continue
        
        # 2. 投影计算
        r_dist = L / (dot_nc + 1e-8)
        xp = r_dist * torch.sum(pn * xn.view(3, 1, 1), dim=0)
        yp = r_dist * torch.sum(pn * yn.view(3, 1, 1), dim=0)

        # 3. 裁剪
        mask_patch = (xp >= -W_p/2.0) & (xp <= W_p/2.0) & \
                     (yp >= -H_p/2.0) & (yp <= H_p/2.0) & mask_front
        
        if not mask_patch.any(): continue

        # 4. 坐标归一化
        grid_u = (xp[mask_patch] / (W_p / 2.0))
        grid_v = -(yp[mask_patch] / (H_p / 2.0)) # 保持图像坐标系定义
        
        # 5. 采样
        curr_patch = patches_tensor[i].unsqueeze(0) 
        grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0).unsqueeze(0)
        sampled_cam_normal = F.grid_sample(curr_patch, grid, mode='bilinear', align_corners=True)
        sampled_cam_normal = sampled_cam_normal.squeeze() 
        
        # --- 保持完全不变的旋转矩阵 ---
        # 既然之前的代码法线颜色是准确的(只错在位置)，那么这个矩阵就是正确的。
        # 它将 Camera Space 的 vector 转换到 Dataset 定义的 World Space。
        R_w2c = torch.stack([xn, -yn, nc], dim=0)
        R_c2w = R_w2c.t()
        
        sampled_world_normal = torch.matmul(R_c2w, sampled_cam_normal)
        
        out_pano[:, mask_patch] += sampled_world_normal
        weight_map[:, mask_patch] += 1.0

    final_pano = out_pano / (weight_map + 1e-8)
    final_pano = F.normalize(final_pano, dim=0, p=2)
    
    return final_pano

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def vis_norm(tensor):
    # tensor: (3, H, W)
    return torch.clamp((tensor + 1.0) / 2.0, 0, 1)

# ==============================================================================
# 3. 主程序
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pano_h', default=1024, type=int)
    parser.add_argument('--pano_w', default=2048, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--resume', default='', type=str, required=True)
    parser.add_argument('--data_path', default='./train', type=str)
    parser.add_argument('--val_data_path', default='./test', type=str)
    parser.add_argument('--output_dir', default='./diagnostic_output_world', type=str)
    parser.add_argument('--model', default=' ', type=str)
    return parser.parse_args()

@torch.no_grad()
def run_diagnostic(model, dataloader, device, args, mode="Train"):
    print(f">>> Running {mode} Diagnostic (World Space Reconstruction)...")
    os.makedirs(os.path.join(args.output_dir, mode), exist_ok=True)
    
    h_fov = 360.0 / (2 * args.grid_height)
    v_fov = 180.0 / args.grid_height

    for i, (views, angles, targets, _) in enumerate(dataloader):
        if i >= 2: break 
        
        views, angles, targets = views.to(device), angles.to(device), targets.to(device)
        preds = model(views, angles).float() # (1, N, 3, H, W)

        # 1. 使用 ODI 逻辑拼接并逆旋转 Prediction
        pano_pred_world = diagnostic_stitch_with_rotation_gpu(
            preds[0], angles[0], args.pano_h, args.pano_w, h_fov, v_fov
        )

        # 2. 使用 ODI 逻辑拼接并逆旋转 Ground Truth (作为验证)
        pano_gt_world = diagnostic_stitch_with_rotation_gpu(
            targets[0], angles[0], args.pano_h, args.pano_w, h_fov, v_fov
        )

        # 3. 准备 RGB (直接拼接，不涉及法线旋转)
        # 这里为了简单直接拼，RGB 不存在坐标系问题
        pano_rgb = torch.zeros((3, args.pano_h, args.pano_w), device=device)
        # ... (可以使用上面的 stitch 函数或简化处理)

        # 绘图对比
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        # 可视化预测 (世界坐标)
        axes[0].imshow(vis_norm(pano_pred_world).permute(1,2,0).cpu().numpy())
        axes[0].set_title(f"{mode} Sample {i}: Prediction (World Space)")
        
        # 可视化真值 (世界坐标)
        axes[1].imshow(vis_norm(pano_gt_world).permute(1,2,0).cpu().numpy())
        axes[1].set_title("GT from Patches (World Space)")
        
        # 加载原始文件做参考
        gt_path = dataloader.dataset.filenames[i]['normal']
        gt_ref = Image.open(gt_path).convert('RGB').resize((args.pano_w, args.pano_h))
        axes[2].imshow(np.array(gt_ref)/255.0)
        axes[2].set_title("Original GT File (Reference)")

        for ax in axes: ax.axis('off')
        plt.savefig(os.path.join(args.output_dir, mode, f"diag_world_{i}.png"), bbox_inches='tight')
        plt.close()

def main():
    args = get_args()
    device = torch.device('cuda')
    
    img_size = args.pano_h // args.grid_height
    model = vit_huge_patch14(img_size=img_size, patch_size=img_size, in_chans=3, 
                             grid_height=args.grid_height, output_size=(args.pano_h, args.pano_w))
    
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()}, strict=False)
    model.to(device).eval()

    train_ds = build_normal_dataset(is_train=True, args=args)
    val_ds = build_normal_dataset(is_train=False, args=args)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    run_diagnostic(model, train_loader, device, args, mode="Train")
    run_diagnostic(model, val_loader, device, args, mode="Val")

if __name__ == '__main__':
    main()