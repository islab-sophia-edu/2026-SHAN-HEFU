import math
import sys
import os
import torch
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
    header = f'Epoch: [{epoch}/{args.epochs}]'
    
    optimizer.zero_grad()

    for data_iter_step, (views, angles, _) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        
        is_last_batch = (data_iter_step == len(data_loader) - 1)
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch
        
        current_mask_ratio = random.uniform(0.6, 0.9) if args.dynamic_mask_ratio else args.mask_ratio
        
        views, angles = views.to(device, non_blocking=True), angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
            # Pass 'angles' to model.forward()
            loss, _, _ = model(views, angles, mask_ratio=current_mask_ratio)

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
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: iter, device: torch.device, epoch: int, args=None):
    '''
    Evaluate model performance on validation set.
    '''
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Evaluate Epoch: [{epoch}]'
    model.eval()

    # Add meter for wMSE (weighted MSE)
    metric_logger.add_meter('wMSE', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    
    data_range = 1.0

    for batch_idx, (views, angles, original_sizes) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        current_mask_ratio = random.uniform(0.6, 0.9) if args.dynamic_mask_ratio else args.mask_ratio
        
        views, angles = views.to(device, non_blocking=True), angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
            # Pass 'angles' to model.forward()
            loss, pred, mask = model(views, angles, mask_ratio=current_mask_ratio)

        metric_logger.update(val_loss=loss.item())

        pred_views = pred.view_as(views)
        
        views_denorm = torch.clamp(views * std + mean, 0, 1)
        
        B, N, C, H, W = views.shape
        # 2.1 将 views (全局归一化空间) 展平以计算 per-patch 统计量
        # 形状: (B, N, C, H, W) -> (B, N, C*H*W)
        views_patches_flat = views.view(B, N, -1) 

        # 2.2 [正确] 计算每个 patch 的均值和标准差 (跨 C, H, W)
        # 我们在最后一个维度 (C*H*W) 上计算
        patch_mean = views_patches_flat.mean(dim=-1, keepdim=True)  # [B, N, 1]
        patch_var = views_patches_flat.var(dim=-1, keepdim=True)    # [B, N, 1]
        patch_std = (patch_var + 1e-6).sqrt()                       # [B, N, 1]

        # 2.3 将 pred_views (per-patch 归一化空间) 展平
        pred_patches_flat = pred_views.view(B, N, -1) # [B, N, C*H*W]

        # 2.4 将 pred_views 从 per-patch 归一化空间转回全局归一化空间
        # 广播机制: [B, N, C*H*W] * [B, N, 1] + [B, N, 1]
        pred_in_global_space_flat = pred_patches_flat * patch_std + patch_mean

        # 2.5 恢复 C, H, W 形状
        pred_in_global_space = pred_in_global_space_flat.view(B, N, C, H, W)

        # 2.6 再用全局参数反归一化到 [0, 1]
        pred_denorm = torch.clamp(pred_in_global_space * std + mean, 0, 1)

        # --- Calculate wMSE (Weighted MSE) for wPSNR ---
        # Use denormalized [0, 1] tensors
        target_flat = views_denorm.view(B, N, -1)
        pred_flat = pred_denorm.view(B, N, -1)
        
        # Per-patch MSE in [0, 1] space
        per_patch_mse = ((pred_flat - target_flat) ** 2).mean(dim=-1) # shape: [B, N]
        
        # Latitude weights
        lat_deg = angles[..., 1] # shape: [B, N]
        weights = torch.cos(torch.deg2rad(lat_deg)).abs() + 1e-6
        
        # L_wMSE_eval = sum(w * mse) / sum(w)
        # Note: We compute wMSE over ALL patches for evaluation
        weighted_mse_sum = (per_patch_mse * weights).sum()
        weight_sum = weights.sum()
        
        # Calculate batch wMSE and update logger
        batch_avg_wmse = weighted_mse_sum / (weight_sum + 1e-6) # Add epsilon
        metric_logger.update(wMSE=batch_avg_wmse.item())
        val_ssim.update(pred_denorm.view(B * N, C, H, W), views_denorm.view(B * N, C, H, W))

        if batch_idx == 0 and misc.is_main_process():
            # Get original pano size and grid parameters
            size_tensor = original_sizes[0]
            pano_width = size_tensor[0].item()
            pano_height = pano_width // 2
            pano_size = (pano_height, pano_width)
            v_steps, u_steps = args.grid_height, 2 * args.grid_height
            h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps

            # --- 1. Create original patch grid ---
            all_patches_np_denorm = [(p.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for p in views_denorm[0]]
            patch_grid_img = create_patch_grid_image(all_patches_np_denorm, u_steps, padding=4)
            
            # Resize grid to match pano width for better visualization
            grid_aspect_ratio = patch_grid_img.height / patch_grid_img.width
            resized_grid_height = int(pano_width * grid_aspect_ratio)
            patch_grid_img = patch_grid_img.resize((pano_width, resized_grid_height), Image.Resampling.LANCZOS)

            # --- 2. Create masked input grid ---
            mask_for_viz = mask[0].cpu().numpy()
            gray_patch_np = np.full_like(all_patches_np_denorm[0], fill_value=128)
            masked_input_patches_list_np = [
                all_patches_np_denorm[i] if mask_for_viz[i] < 0.5 else gray_patch_np
                for i in range(len(all_patches_np_denorm))
            ]
            masked_grid_img = create_patch_grid_image(masked_input_patches_list_np, u_steps, padding=4)
            masked_grid_img = masked_grid_img.resize(patch_grid_img.size, Image.Resampling.NEAREST)

            # --- Get raw reconstruction patches ---
            reconstructed_patches_np_denorm = [(p.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for p in pred_denorm[0]]

            # --- 3. Create blended reconstruction grid (NEW) ---
            # This grid uses:
            # - Original patch (all_patches_np_denorm) if VISIBLE (mask < 0.5)
            # - Reconstructed patch (reconstructed_patches_np_denorm) if MASKED (mask >= 0.5)
            blended_reconstruction_patches_np = [
                all_patches_np_denorm[i] if mask_for_viz[i] < 0.5 else reconstructed_patches_np_denorm[i]
                for i in range(len(all_patches_np_denorm))
            ]
            reconstructed_grid_img = create_patch_grid_image(blended_reconstruction_patches_np, u_steps, padding=4)
            reconstructed_grid_img = reconstructed_grid_img.resize(patch_grid_img.size, Image.Resampling.LANCZOS)

            # --- 4. Define stitching helper --- (Renumbered)
            def stitch_pano_from_list(patches_np_list_input):
                patches_float = [p.astype(np.float32) / 255.0 for p in patches_np_list_input]
                pano_np = odi_helper.embed_patches_to_pano(
                    patches_np_list=patches_float, pano_size=pano_size,
                    h_num=u_steps, v_num=v_steps, h_fov_deg=h_fov, v_fov_deg=v_fov
                )
                return (pano_np * 255).astype(np.uint8)

            # --- 5. Create blended and original panoramas --- (Renumbered)
            # Use the same blended list from step 3 for the reconstructed pano
            #masking patches decoder output + original visible patches
            #reconstructed_pano_img = Image.fromarray(stitch_pano_from_list(blended_reconstruction_patches_np))
            #all decoder outputs for image
            reconstructed_pano_img = Image.fromarray(stitch_pano_from_list(reconstructed_patches_np_denorm))
            # Original pano
            original_pano_img = Image.fromarray(stitch_pano_from_list(all_patches_np_denorm))

            # --- 6. Combine all images and save --- (Renumbered)
            final_img_width = pano_width
            # We now have 3 grids + 2 panos
            final_img_height = (resized_grid_height * 3) + (original_pano_img.height * 2)
            combined_img = Image.new('RGB', (final_img_width, final_img_height))
            
            y_offset = 0
            # Image 1: Original Patches
            combined_img.paste(patch_grid_img, (0, y_offset)); y_offset += patch_grid_img.height
            # Image 2: Masked Input
            combined_img.paste(masked_grid_img, (0, y_offset)); y_offset += masked_grid_img.height
            # Image 3: Blended Reconstruction (NEW)
            combined_img.paste(reconstructed_grid_img, (0, y_offset)); y_offset += reconstructed_grid_img.height 
            # Image 4: Blended Reconstructed Pano
            combined_img.paste(reconstructed_pano_img, (0, y_offset)); y_offset += reconstructed_pano_img.height
            # Image 5: Original Pano
            combined_img.paste(original_pano_img, (0, y_offset))

            save_path = os.path.join(args.output_dir, f'diagnostic_epoch_{epoch}.png')
            combined_img.save(save_path)
            print(f"Saved combined diagnostic visualization to {save_path}")
            
    # --- Compute final metrics ---
    metric_logger.synchronize_between_processes()
    
    # Get global average wMSE from logger
    global_avg_wmse = metric_logger.meters['wMSE'].global_avg
    
    # Calculate wPSNR from global average wMSE
    if global_avg_wmse > 0:
        wpsnr_value = 10 * math.log10((data_range**2) / global_avg_wmse)
    else:
        wpsnr_value = float('inf') 
        
    ssim_value = val_ssim.compute().item()
    
    # Update logger with final computed values
    metric_logger.add_meter('wPSNR', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.update(wPSNR=wpsnr_value, ssim=ssim_value)

    print('Averaged validation stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}