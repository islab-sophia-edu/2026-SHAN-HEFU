import math
import sys
import os
from xml.parsers.expat import model
import torch
import torch.nn.functional as F

import torchvision
import random
import numpy as np
from PIL import Image

import util.misc as misc
import torchmetrics
import util.odi_processing as odi_helper

def create_patch_grid_image(patches_list, grid_width_in_patches, padding=4, bg_color=(0, 0, 0)):
    """Creates a grid image of all patches with spacing between them."""
    if not patches_list: return None
    
    first_patch_uint8 = patches_list[0]
    if first_patch_uint8.dtype != np.uint8:
        if first_patch_uint8.max() <= 1.0: first_patch_uint8 = (first_patch_uint8 * 255).astype(np.uint8)
        else: first_patch_uint8 = first_patch_uint8.astype(np.uint8)
    patch_height, patch_width, channels = first_patch_uint8.shape
    
    num_patches = len(patches_list)
    grid_height_in_patches = (num_patches + grid_width_in_patches - 1) // grid_width_in_patches

    canvas_height = grid_height_in_patches * (patch_height + padding) + padding
    canvas_width = grid_width_in_patches * (patch_width + padding) + padding
    
    canvas = np.full((canvas_height, canvas_width, channels), bg_color, dtype=np.uint8)

    for i, patch in enumerate(patches_list):
        row = i // grid_width_in_patches
        col = i % grid_width_in_patches
        y_start = padding + row * (patch_height + padding)
        x_start = padding + col * (patch_width + padding)
        
        patch_uint8 = patch
        if patch_uint8.dtype != np.uint8:
            if patch_uint8.max() <= 1.0: patch_uint8 = (patch_uint8 * 255).astype(np.uint8)
            else: patch_uint8 = patch_uint8.astype(np.uint8)
                
        canvas[y_start : y_start + patch_height, x_start : x_start + patch_width] = patch_uint8
        
    return Image.fromarray(canvas)

def train_one_epoch(model: torch.nn.Module, data_loader: iter, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device, epoch: int,
                    loss_scaler, args=None):
    """
    Train model for one epoch.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    
    # [新增] 监控 RGB 和 XYZ 的 Loss
    metric_logger.add_meter('loss_rgb', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_xyz', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    
    header = f'Epoch: [{epoch}/{args.epochs}]'
    optimizer.zero_grad()

    for data_iter_step, (views, angles, _) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        
        is_last_batch = (data_iter_step == len(data_loader) - 1)
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch
        
        current_mask_ratio = random.uniform(0.6, 0.9) if args.dynamic_mask_ratio else args.mask_ratio
        
        views, angles = views.to(device, non_blocking=True), angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
            loss, _, _, loss_rgb, loss_xyz = model(views, angles, mask_ratio=current_mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        
        loss_scaler(
            loss, 
            optimizer, 
            clip_grad=args.clip_grad, 
            parameters=model.parameters(),
            update_grad=is_update_step
        )

        if is_update_step:
            optimizer.zero_grad()
            scheduler.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_rgb=loss_rgb.item())
        metric_logger.update(loss_xyz=loss_xyz.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: iter, device: torch.device, epoch: int, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Evaluate Epoch: [{epoch}]'
    model.eval()

    # 仅在 RGB 通道上计算指标
    val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # ImageNet Mean/Std (仅用于 RGB)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    
    for batch_idx, (views, angles, original_sizes) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        current_mask_ratio = random.uniform(0.6, 0.9) if args.dynamic_mask_ratio else args.mask_ratio
        
        views, angles = views.to(device, non_blocking=True), angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
            loss, pred, mask, _, _ = model(views, angles, mask_ratio=current_mask_ratio)

        metric_logger.update(val_loss=loss.item())

        # --- [修复开始] 数据拆分与处理 ---
        B, N, C, H, W = views.shape
        pred_views = pred.view_as(views)

        # 1. 拆分 RGB 和 XYZ
        views_rgb = views[:, :, :3, :, :]   # GT RGB
        views_xyz = views[:, :, 3:, :, :]   # GT XYZ
        
        pred_rgb = pred_views[:, :, :3, :, :] # Pred RGB
        pred_xyz = pred_views[:, :, 3:, :, :] # Pred XYZ

        # 2. RGB 反归一化 (用于指标计算和可视化)
        # GT Denorm
        views_rgb_denorm = torch.clamp(views_rgb * std + mean, 0, 1)
        
        # Pred Denorm (需要先恢复方差)
        # 计算当前 Batch 的 Patch 统计量来恢复预测值的分布 (MAE 标准做法)
        # 注意：这里我们只对 RGB 做这个操作
        target_rgb_flat = views_rgb.reshape(B, N, -1)
        mean_flt = target_rgb_flat.mean(dim=-1, keepdim=True)
        var_flt = target_rgb_flat.var(dim=-1, keepdim=True)
        std_flt = (var_flt + 1e-6).sqrt()
        
        pred_rgb_flat = pred_rgb.reshape(B, N, -1)
        pred_rgb_global = pred_rgb_flat * std_flt + mean_flt
        pred_rgb_global = pred_rgb_global.view(B, N, 3, H, W)
        pred_rgb_denorm = torch.clamp(pred_rgb_global * std + mean, 0, 1)

        # 3. 计算指标 (仅 RGB)
        # Flatten B*N patches
        val_psnr.update(pred_rgb_denorm.reshape(-1, 3, H, W), views_rgb_denorm.reshape(-1, 3, H, W))
        val_ssim.update(pred_rgb_denorm.reshape(-1, 3, H, W), views_rgb_denorm.reshape(-1, 3, H, W))

        # --- Diagnostic Visualization (Fully GPU Optimized) ---
        if batch_idx == 0 and misc.is_main_process():
            # 取第一个样本，且只取 RGB 用于可视化
            ori_patches = views_rgb_denorm[0]  
            rec_patches = pred_rgb_denorm[0] 
            mask_patches = mask[0]
            angles_gpu = angles[0]
            
            # 参数获取
            pano_h = getattr(args, 'pano_h', 512) 
            pano_w = getattr(args, 'pano_w', 1024)
            v_steps = args.grid_height
            u_steps = 2 * args.grid_height
            h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps

            # Grid 1: Original Patches
            grid_img_ori = odi_helper.make_grid_gpu(ori_patches, grid_width=u_steps)
            
            # Grid 2: Masked Input (Gray out masked areas)
            gray_tensor = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1).expand_as(ori_patches[0])
            gray_tensor = gray_tensor.unsqueeze(0) # (1, 3, H, W)
            
            mask_expanded = mask_patches.view(N, 1, 1, 1)
            masked_input_patches = torch.where(mask_expanded > 0.5, gray_tensor, ori_patches)
            grid_img_masked = odi_helper.make_grid_gpu(masked_input_patches, grid_width=u_steps)
            
            # Grid 3: Blended Reconstruction Patches
            blended_patches = torch.where(mask_expanded > 0.5, rec_patches, ori_patches)
            grid_img_blend = odi_helper.make_grid_gpu(blended_patches, grid_width=u_steps)
            
            # Stitching to Pano (GPU) - 仅使用 RGB Patch
            pano_rec = odi_helper.embed_patches_to_pano_gpu(
                rec_patches, pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu
            )
            
            pano_ori = odi_helper.embed_patches_to_pano_gpu(
                ori_patches, pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu
            )

            # Resize Grids to match Pano Width
            def resize_tensor(img_tensor, target_w):
                C, H, W = img_tensor.shape
                scale = target_w / W
                target_h = int(H * scale)
                img_unsqueezed = img_tensor.unsqueeze(0)
                resized = F.interpolate(img_unsqueezed, size=(target_h, target_w), mode='bilinear', align_corners=False)
                return resized.squeeze(0)
            
            # 确保 grid 是 3通道 (od_helper如果处理得当应该没问题)
            grid_img_ori_res = resize_tensor(grid_img_ori, pano_w)
            grid_img_masked_res = resize_tensor(grid_img_masked, pano_w)
            grid_img_blend_res = resize_tensor(grid_img_blend, pano_w)
            
            # Concatenate Vertically
            final_tensor = torch.cat([
                grid_img_ori_res,
                grid_img_masked_res,
                grid_img_blend_res,
                pano_rec, 
                pano_ori
            ], dim=1)
            
            save_path = os.path.join(args.output_dir, f'diagnostic_epoch_{epoch}.png')
            torchvision.utils.save_image(final_tensor, save_path)

    # ... [Rest is same] ...
    psnr_value = val_psnr.compute().item()
    ssim_value = val_ssim.compute().item()
    
    metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.update(psnr=psnr_value, ssim=ssim_value)

    print('Averaged validation stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}