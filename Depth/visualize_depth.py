import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import util.misc as misc
import util.odi_processing_gpu as odi_helper
from dataset_stanford_depth import build_depth_dataset

# 导入模型
try:
    from models_depth_cnn import vit_huge_patch14
except ImportError:
    # 尝试兼容其他文件名
    try:
        from models_normal_cnn import vit_huge_patch14
        print("Loaded vit_huge_patch14 from models_normal_cnn")
    except ImportError:
        print("Error: Could not import 'vit_huge_patch14'. Please check your model filename.")
        sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser('Pano Depth Visualization', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--pano_h', default=512, type=int, help='Pano Height')
    parser.add_argument('--pano_w', default=1024, type=int, help='Pano Width')
    parser.add_argument('--grid_height', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    # Model params
    parser.add_argument('--model', default='vit_huge_patch14', type=str)
    parser.add_argument('--resume', default='', type=str, required=True, help='Path to checkpoint')
    
    # Data params
    parser.add_argument('--val_data_path', default='./test', type=str)
    parser.add_argument('--output_dir', default='./vis_depth_output', type=str)
    
    # Visualization params
    parser.add_argument('--num_samples', default=5, type=int, help='Number of images to save')
    parser.add_argument('--max_depth', default=10.0, type=float, help='Max depth for visualization clipping')
    
    return parser.parse_args()

def colorize_depth(depth_tensor, max_val=10.0, cmap_name='magma'):
    """
    Input: Tensor [1, H, W] or [H, W], range [0, max_val]
    Output: Tensor [3, H, W], range [0, 1]
    """
    if depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.squeeze(0)
    
    # Normalize to [0, 1]
    norm_depth = torch.clamp(depth_tensor / max_val, 0, 1)
    
    # To Numpy for colormap
    norm_depth_np = norm_depth.cpu().numpy()
    
    cm = plt.get_cmap(cmap_name)
    colorized = cm(norm_depth_np)[:, :, :3] 
    
    colorized_tensor = torch.from_numpy(colorized).permute(2, 0, 1).float()
    return colorized_tensor.to(depth_tensor.device)

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. 计算 Patch Size
    img_size = args.pano_h // args.grid_height
    print(f"Visualization Settings: Pano {args.pano_w}x{args.pano_h}, Grid H={args.grid_height}, Patch={img_size}")

    # 2. Dataset & DataLoader
    dataset_val = build_depth_dataset(is_train=False, args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, 
        shuffle=False, num_workers=0, pin_memory=False
    )

    # 3. Model Setup
    print(f"Loading model from {args.resume}...")
    
    # [修复1] 显式传递 patch_size，防止默认值覆盖
    model = globals()[args.model](
        img_size=img_size,
        patch_size=img_size, # <--- 关键修复：确保卷积核大小等于 Patch 大小
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
    )

    # 加载权重
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model_dict = model.state_dict()
    
    # --- [修复2] 权重自动插值 (Interpolation) ---
    # 扫描所有 view_embed 相关权重，如果形状不匹配，自动插值
    for k in list(state_dict.keys()):
        if 'view_embed' in k and 'weight' in k and k in model_dict:
            ckpt_shape = state_dict[k].shape
            model_shape = model_dict[k].shape
            
            if ckpt_shape != model_shape:
                print(f"Interpolating {k}: {ckpt_shape} -> {model_shape}")
                # F.interpolate 需要 (N, C, H, W)
                # 权重通常是 (Out, In, kH, kW)
                interpolated_weight = F.interpolate(
                    state_dict[k], 
                    size=model_shape[2:], # Target H, W
                    mode='bicubic', 
                    align_corners=False
                )
                state_dict[k] = interpolated_weight

    # --- [修复3] Head 通道修正 ---
    # 自动检测 checkpoint 中的输出通道数
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

    # 加载处理后的权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded. Missing keys: {len(msg.missing_keys)}")
    
    model.to(device)
    model.eval()

    # 4. Visualization Loop
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Pano Params
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    h_fov = 360.0 / u_steps
    v_fov = 180.0 / v_steps

    print(f"Generating {args.num_samples} samples...")

    for i, (views, angles, targets) in enumerate(data_loader):
        if i >= args.num_samples:
            break
            
        views = views.to(device)   
        angles = angles.to(device) 
        targets = targets.to(device)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            preds = model(views, angles)
        
        preds = preds.float()

        # Interpolate preds if needed
        if preds.shape[-1] != views.shape[-1]:
             B, N, C, H_out, W_out = preds.shape
             preds = F.interpolate(
                 preds.view(B*N, C, H_out, W_out), 
                 size=(img_size, img_size), 
                 mode='bilinear', align_corners=False
             ).view(B, N, C, img_size, img_size)

        idx = 0
        patches_rgb = views[idx]
        patches_pred = preds[idx]
        patches_gt = targets[idx]
        sample_angles = angles[idx]

        # Stitching
        patches_rgb_denorm = torch.clamp(patches_rgb * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1), 0, 1)
        pano_rgb = odi_helper.embed_patches_to_pano_gpu(
            patches_rgb_denorm, args.pano_h, args.pano_w, 
            u_steps, v_steps, h_fov, v_fov, sample_angles
        )

        pano_pred_raw = odi_helper.embed_patches_to_pano_gpu(
            patches_pred, args.pano_h, args.pano_w,
            u_steps, v_steps, h_fov, v_fov, sample_angles
        )
        pano_pred_vis = colorize_depth(pano_pred_raw, max_val=args.max_depth)

        pano_gt_raw = odi_helper.embed_patches_to_pano_gpu(
            patches_gt, args.pano_h, args.pano_w,
            u_steps, v_steps, h_fov, v_fov, sample_angles
        )
        pano_gt_vis = colorize_depth(pano_gt_raw, max_val=args.max_depth)

        # Fix small dimension mismatch
        if pano_rgb.shape[1:] != pano_pred_vis.shape[1:]:
             target_h, target_w = pano_rgb.shape[1], pano_rgb.shape[2]
             pano_pred_vis = F.interpolate(pano_pred_vis.unsqueeze(0), (target_h, target_w), mode='nearest').squeeze(0)
             pano_gt_vis = F.interpolate(pano_gt_vis.unsqueeze(0), (target_h, target_w), mode='nearest').squeeze(0)

        combined = torch.cat([pano_rgb, pano_pred_vis, pano_gt_vis], dim=1)
        
        save_path = os.path.join(args.output_dir, f'sample_{i}.png')
        from torchvision.utils import save_image
        save_image(combined, save_path)
        print(f"Saved {save_path}")

    print("Done!")

if __name__ == '__main__':
    main()