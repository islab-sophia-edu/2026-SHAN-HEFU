import math
import sys
import os
import torch
import torch.nn.functional as F

import torchvision
import numpy as np
from PIL import Image

import util.misc as misc
import torchmetrics
import util.odi_processing as odi_helper


def create_patch_grid_image(patches_list, grid_width_in_patches, padding=4, bg_color=(0, 0, 0)):
    """Creates a grid image of all patches with spacing between them."""
    if not patches_list:
        return None

    first_patch_uint8 = patches_list[0]
    if first_patch_uint8.dtype != np.uint8:
        if first_patch_uint8.max() <= 1.0:
            first_patch_uint8 = (first_patch_uint8 * 255).astype(np.uint8)
        else:
            first_patch_uint8 = first_patch_uint8.astype(np.uint8)
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
            if patch_uint8.max() <= 1.0:
                patch_uint8 = (patch_uint8 * 255).astype(np.uint8)
            else:
                patch_uint8 = patch_uint8.astype(np.uint8)

        canvas[y_start: y_start + patch_height, x_start: x_start + patch_width] = patch_uint8

    return Image.fromarray(canvas)


def _unpack_batch(batch):
    """
    Backward-compatible batch unpacking.

    Old dataset returns: views, angles, weights
    Oriented-SPE / GCTT dataset returns: views, angles, gauge_angles, weights
    """
    if len(batch) == 4:
        views, angles, gauge_angles, weights = batch
    elif len(batch) == 3:
        views, angles, weights = batch
        gauge_angles = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}. Expected 3 or 4.")
    return views, angles, gauge_angles, weights


def train_one_epoch(model: torch.nn.Module, data_loader: iter, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device, epoch: int,
                    loss_scaler, args=None):
    """Train model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        views, angles, gauge_angles, _ = _unpack_batch(batch)

        is_last_batch = (data_iter_step == len(data_loader) - 1)
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch

        # NoAdaptive ablation: fixed MAE random mask ratio.
        # Do not sample a dynamic ratio during training.
        current_mask_ratio = 0.75

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
            loss, _, _ = model(
                views,
                angles,
                gauge_angles=gauge_angles,
                mask_ratio=current_mask_ratio,
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / args.accum_iter

        loss_scaler(
            loss,
            optimizer,
            clip_grad=args.clip_grad,
            parameters=model.parameters(),
            update_grad=is_update_step,
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
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Evaluate Epoch: [{epoch}]'
    model.eval()

    val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views, angles, gauge_angles, weights = _unpack_batch(batch)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
            loss, pred, mask = model(
                views,
                angles,
                gauge_angles=gauge_angles,
                mask_ratio=0.75,
            )

        metric_logger.update(val_loss=loss.item())

        pred_views = pred.view_as(views)
        B, N, C, H, W = views.shape

        if getattr(args, 'norm_pix_loss', False):
            views_patches_flat = views.view(B, N, -1)
            patch_mean = views_patches_flat.mean(dim=-1, keepdim=True)
            patch_var = views_patches_flat.var(dim=-1, keepdim=True)
            patch_std = (patch_var + 1e-6).sqrt()
            pred_patches_flat = pred_views.view(B, N, -1)
            pred_in_global_space = (pred_patches_flat * patch_std + patch_mean).view(B, N, C, H, W)
        else:
            pred_in_global_space = pred_views

        pred_denorm = torch.clamp(pred_in_global_space * std + mean, 0, 1)
        views_denorm = torch.clamp(views * std + mean, 0, 1)

        val_psnr.update(pred_denorm.view(B * N, C, H, W), views_denorm.view(B * N, C, H, W))
        val_ssim.update(pred_denorm.view(B * N, C, H, W), views_denorm.view(B * N, C, H, W))

        if batch_idx == 0 and misc.is_main_process():
            ori_patches = views_denorm[0]
            rec_patches = pred_denorm[0]
            mask_patches = mask[0]
            angles_gpu = angles[0]

            pano_h = getattr(args, 'pano_h', 512)
            pano_w = getattr(args, 'pano_w', 1024)
            v_steps = args.grid_height
            u_steps = 2 * args.grid_height
            h_fov, v_fov = 360.0 / u_steps, 180.0 / v_steps

            grid_img_ori = odi_helper.make_grid_gpu(ori_patches, grid_width=u_steps)

            gray_tensor = torch.full_like(ori_patches[0:1], 0.5)
            mask_expanded = mask_patches.view(N, 1, 1, 1)
            masked_input_patches = torch.where(mask_expanded > 0.5, gray_tensor, ori_patches)
            grid_img_masked = odi_helper.make_grid_gpu(masked_input_patches, grid_width=u_steps)

            blended_patches = torch.where(mask_expanded > 0.5, rec_patches, ori_patches)
            grid_img_blend = odi_helper.make_grid_gpu(blended_patches, grid_width=u_steps)

            pano_rec = odi_helper.embed_patches_to_pano_gpu(
                rec_patches, pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu
            )
            pano_ori = odi_helper.embed_patches_to_pano_gpu(
                ori_patches, pano_h, pano_w, u_steps, v_steps, h_fov, v_fov, angles_gpu
            )

            def resize_tensor(img_tensor, target_w):
                C_, H_, W_ = img_tensor.shape
                scale = target_w / W_
                target_h = int(H_ * scale)
                resized = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )
                return resized.squeeze(0)

            grid_img_ori_res = resize_tensor(grid_img_ori, pano_w)
            grid_img_masked_res = resize_tensor(grid_img_masked, pano_w)
            grid_img_blend_res = resize_tensor(grid_img_blend, pano_w)

            final_tensor = torch.cat([
                grid_img_ori_res,
                grid_img_masked_res,
                grid_img_blend_res,
                pano_rec,
                pano_ori,
            ], dim=1)

            save_path = os.path.join(args.output_dir, f'diagnostic_epoch_{epoch}.png')
            torchvision.utils.save_image(final_tensor, save_path)

    psnr_value = val_psnr.compute().item()
    ssim_value = val_ssim.compute().item()

    metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.update(psnr=psnr_value, ssim=ssim_value)

    print('Averaged validation stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
