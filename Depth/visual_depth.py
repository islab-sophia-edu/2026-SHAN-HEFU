import argparse
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt


import util.misc as misc
import util.odi_processing_gpu as odi_helper 
from dataset_stanford_depth import build_depth_dataset

try:
    from models_depth_cnn import vit_huge_patch14
except ImportError:
    from models_normal_cnn import vit_huge_patch14

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Depth Top-10 Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model', default='vit_huge_patch14', type=str)
    parser.add_argument('--pano_h', default=1024, type=int)
    parser.add_argument('--pano_w', default=2048, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--data_path', default='/home/shanhefu/Stanford2D3D', type=str)
    parser.add_argument('--val_data_path', default='/home/shanhefu/Stanford2D3D', type=str)
    parser.add_argument('--resume', default='/media/data_hdd1/shanhefu/outputs/finetune/depth_stanford/depth_mask0.6-0.9_16*32_3.2e-3_decay_huge_1024_cnn_16_4_l1_hybridall_test2/checkpoint-99.pth', help='Path to checkpoint')
    parser.add_argument('--output_dir', default='/media/data_hdd1/shanhefu/outputs/comparison/Depth_visual_gap', help='Path to save results')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--input_size', default=None) 
    
    parser.add_argument('--max_depth', default=100.0, type=float) 
    return parser

# ==========================================
# [完美修复] 基于百分位数的伪彩色映射函数
# ==========================================
def colorize_depth_np(depth_np, valid_mask=None, cmap=cv2.COLORMAP_INFERNO):
    # 如果没有传入 mask，就提取当前图里大于 0 的区域
    if valid_mask is None:
        valid_mask = depth_np > 1e-3
        
    if not valid_mask.any():
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
        
    valid_pixels = depth_np[valid_mask]
    
    # [核心拉伸逻辑] 过滤掉前 2% 的极近点和后 2% 的极远点 (如 100m 离群值)
    # 这不是造假，这是标准的动态对比度拉伸，完美解决死黑问题
    vmin = np.percentile(valid_pixels, 2)
    vmax = np.percentile(valid_pixels, 98)
    
    if vmax - vmin > 1e-5:
        norm_depth = (depth_np - vmin) / (vmax - vmin)
    else:
        norm_depth = np.zeros_like(depth_np)
        
    norm_depth = np.clip(norm_depth, 0, 1)
    depth_8bit = (norm_depth * 255.0).astype(np.uint8)
    
    colorized = cv2.applyColorMap(depth_8bit, cmap)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    # 保持背景为纯黑
    colorized[~valid_mask] = 0
    
    return colorized

def compute_rmse(pred, target):
    """计算 Root Mean Square Error"""
    valid_mask = target > 1e-3 
    if valid_mask.sum() == 0:
        return float('inf')
    
    diff = pred[valid_mask] - target[valid_mask]
    mse = (diff ** 2).mean()
    return np.sqrt(mse)

@torch.no_grad()
def run_top10_depth_evaluation(data_loader, dataset, model, device, args):
    model.eval()
    results_list = []
    
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)
    
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    h_fov = 360.0 / u_steps
    v_fov = 180.0 / v_steps

    file_list = dataset.filenames 
    total_samples_to_scan = min(143, len(file_list)) 

    print(f"--- Phase 1: Scanning top {total_samples_to_scan} test samples for RMSE ---")
    for batch_idx, (views, angles, targets) in enumerate(data_loader):
        views, angles, targets = views.to(device), angles.to(device), targets.to(device)
        B, N, C = views.shape[:3]
        
        global_idx_start = batch_idx * args.batch_size
        if global_idx_start >= total_samples_to_scan:
            break

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(views, angles) 
            # [修正] 约束预测结果不出现负数
            preds = torch.relu(preds)
            
            preds_up = F.interpolate(
                preds.view(B*N, 1, preds.shape[-2], preds.shape[-1]),
                size=(patch_h, patch_w), mode='bilinear', align_corners=False
            )
            targets_up = F.interpolate(
                targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]),
                size=(patch_h, patch_w), mode='nearest'
            )

        for b in range(B):
            global_idx = global_idx_start + b
            if global_idx >= total_samples_to_scan:
                break
                
            sample_pred = preds_up[b * N : (b+1) * N].cpu().numpy().squeeze() * 100.0
            sample_gt = targets_up[b * N : (b+1) * N].cpu().numpy().squeeze() * 100.0
            
            rmse_val = compute_rmse(sample_pred, sample_gt)
            fname = os.path.basename(file_list[global_idx]['rgb'])
            
            results_list.append({
                'index': global_idx, 
                'fname': fname, 
                'rmse': rmse_val
            })
            
        if (batch_idx + 1) % 5 == 0:
            print(f"Scanned {global_idx_start + B}/{total_samples_to_scan} samples")

    df = pd.DataFrame(results_list)
    top_10 = df.sort_values(by='rmse', ascending=True).head(10).reset_index(drop=True)
    target_indices = set(top_10['index'].values)
    
    print(f"\n--- Top 10 Best Depth Predictions Identified ---")
    print(top_10)

    print(f"\n--- Phase 3: Generating Visualizations for Top 10 ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    rank_map = {row['index']: (idx+1, row['fname']) for idx, row in top_10.iterrows()}

    for batch_idx, (views, angles, targets) in enumerate(data_loader):
        B = views.size(0)
        for b in range(B):
            global_idx = batch_idx * args.batch_size + b
            
            if global_idx in target_indices:
                rank, fname = rank_map[global_idx]
                rmse_val = top_10.loc[top_10['index'] == global_idx, 'rmse'].values[0]
                print(f"Rendering Rank {rank:02d} (File: {fname}, RMSE: {rmse_val:.4f})...")
                
                v_t = views[b:b+1].to(device)
                a_t = angles[b].to(device)
                t_t = targets[b:b+1].to(device)
                N = v_t.size(1)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(v_t, angles[b:b+1].to(device))
                    print(f"Preds Min: {preds.min().item():.4f}, Max: {preds.max().item():.4f}, Mean: {preds.mean().item():.4f}, Std: {preds.std().item():.4f}")
                    # [修复点] 必须加 relu 去除负数
                    preds = torch.relu(preds) 
                    
                    preds_tensor = F.interpolate(
                        preds.view(N, 1, preds.shape[-2], preds.shape[-1]),
                        size=(patch_h, patch_w), mode='bilinear', align_corners=False
                    ).float() 
                
                t_flat = t_t.view(N, 1, t_t.shape[-2], t_t.shape[-1]).float()
                gt_tensor = F.interpolate(t_flat, size=(patch_h, patch_w), mode='nearest')
                
                rgb_patches = (v_t[0] * std + mean).clamp(0, 1)

                pano_rgb_t = odi_helper.embed_patches_to_pano_gpu(
                    rgb_patches, args.pano_h, args.pano_w, 
                    u_steps, v_steps, h_fov, v_fov, a_t
                )
                pano_rgb_img = (pano_rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                pano_pred_t = odi_helper.embed_patches_to_pano_gpu(
                    preds_tensor, args.pano_h, args.pano_w, 
                    u_steps, v_steps, h_fov, v_fov, a_t
                )
                pano_gt_t = odi_helper.embed_patches_to_pano_gpu(
                    gt_tensor, args.pano_h, args.pano_w, 
                    u_steps, v_steps, h_fov, v_fov, a_t
                )

                pano_pred_np = pano_pred_t.squeeze().cpu().numpy() * 100.0
                pano_gt_np = pano_gt_t.squeeze().cpu().numpy() * 100.0

                # [完美修复] 统一使用 GT 的 Mask！
                # 这样预测的图即使有偏差，它的墙角/天花板背景黑边也和 GT 保持完美一致
                pano_gt_mask = pano_gt_np > 1e-3
                
                pano_gt_color = colorize_depth_np(pano_gt_np, valid_mask=pano_gt_mask)
                pano_pred_color = colorize_depth_np(pano_pred_np, valid_mask=pano_gt_mask)

                base_name = os.path.splitext(fname)[0].replace('_rgb', '') 
                folder_name = f"{base_name}_Rank{rank}"
                save_folder = os.path.join(args.output_dir, folder_name)
                os.makedirs(save_folder, exist_ok=True)

                Image.fromarray(pano_rgb_img).save(os.path.join(save_folder, f"{base_name}_Rank{rank}_RGB.png"))
                Image.fromarray(pano_gt_color).save(os.path.join(save_folder, f"{base_name}_Rank{rank}_GT.png"))
                Image.fromarray(pano_pred_color).save(os.path.join(save_folder, f"{base_name}_Rank{rank}_Predicted.png"))
                
                with open(os.path.join(save_folder, "info.txt"), "w") as info_file:
                    info_file.write(f"File Name: {fname}\n")
                    info_file.write(f"Global Index: {global_idx}\n")
                    info_file.write(f"Rank: {rank}\n")
                    info_file.write(f"RMSE: {rmse_val:.4f}\n")
                
                target_indices.remove(global_idx)
                
        if not target_indices:
            break

    print(f"\n[✔] Depth evaluation and visualization completed at: {args.output_dir}")

def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)
    real_patch_size = (patch_h, patch_w)
    args.input_size = real_patch_size

    dataset_val = build_depth_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False, shuffle=False
    )

    print(f"Creating Depth Model: {args.model}")
    model = vit_huge_patch14(
        img_size=real_patch_size,
        patch_size=real_patch_size,
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w)
    )

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model_dict = model.state_dict()
    
    for k in list(state_dict.keys()):
        if 'view_embed' in k and 'weight' in k and k in model_dict:
            ckpt_shape = state_dict[k].shape
            model_shape = model_dict[k].shape
            
            if ckpt_shape != model_shape:
                print(f"Interpolating {k}: {ckpt_shape} -> {model_shape}")
                interpolated_weight = F.interpolate(
                    state_dict[k], 
                    size=model_shape[2:], 
                    mode='bicubic', 
                    align_corners=False
                )
                state_dict[k] = interpolated_weight

    head_modules = list(model.head.named_children())
    last_conv_name = None
    last_conv_idx = -1
    
    for name, module in reversed(head_modules):
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
            last_conv_idx = int(name)
            break
            
    if last_conv_name is not None:
        ckpt_key_weight = f'head.{last_conv_name}.weight'
        if ckpt_key_weight in state_dict:
            ckpt_out_chans = state_dict[ckpt_key_weight].shape[0]
            model_out_chans = model.head[last_conv_idx].out_channels
            
            if ckpt_out_chans != model_out_chans:
                print(f"Fixing Head Channels: {model_out_chans} -> {ckpt_out_chans}")
                old_layer = model.head[last_conv_idx]
                new_layer = nn.Conv2d(
                    in_channels=old_layer.in_channels,
                    out_channels=ckpt_out_chans,
                    kernel_size=old_layer.kernel_size,
                    stride=old_layer.stride,
                    padding=old_layer.padding,
                    bias=(old_layer.bias is not None)
                )
                model.head[last_conv_idx] = new_layer

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded. Missing keys: {len(msg.missing_keys)}")
    
    model.to(device)

    run_top10_depth_evaluation(data_loader_val, dataset_val, model, device, args)

if __name__ == '__main__':
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)