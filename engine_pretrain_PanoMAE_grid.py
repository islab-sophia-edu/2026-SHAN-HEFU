import math
import os
import sys
import random

import torch
import torch.nn.functional as F
import torchvision
import torchmetrics

import util.misc as misc


def _unpack_batch(batch):
    """Accept both PB-style and GCTT-compatible batches."""
    if len(batch) == 4:
        views, angles, gauge_angles, weights = batch
    elif len(batch) == 3:
        views, angles, weights = batch
        gauge_angles = None
    else:
        raise ValueError(f'Unexpected batch length {len(batch)}. Expected 3 or 4.')
    return views, angles, gauge_angles, weights


def _unpatchify_grid(patches: torch.Tensor, grid_height: int) -> torch.Tensor:
    """Recompose row-major ERP grid patches.

    Args:
        patches: [N, C, P, P], row-major order with u_steps=2*grid_height.
        grid_height: number of vertical grid rows.
    Returns:
        image: [C, grid_height*P, 2*grid_height*P]
    """
    if patches.dim() != 4:
        raise ValueError(f'Expected [N,C,P,P], got {tuple(patches.shape)}')
    v_steps = int(grid_height)
    u_steps = 2 * v_steps
    N, C, P_h, P_w = patches.shape
    if P_h != P_w:
        raise ValueError(f'Expected square patches, got {P_h}x{P_w}.')
    if N != v_steps * u_steps:
        raise ValueError(f'Expected {v_steps * u_steps} patches, got {N}.')
    return (
        patches
        .view(v_steps, u_steps, C, P_h, P_w)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .view(C, v_steps * P_h, u_steps * P_w)
    )


def _resize_to_width(img_tensor: torch.Tensor, target_w: int) -> torch.Tensor:
    C, H, W = img_tensor.shape
    if W == target_w:
        return img_tensor
    target_h = int(round(H * target_w / W))
    return F.interpolate(
        img_tensor.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)


def train_one_epoch(model: torch.nn.Module, data_loader: iter, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device, epoch: int,
                    loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'

    optimizer.zero_grad()

    device_type = device.type
    autocast_enabled = device_type == 'cuda'

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        views, angles, gauge_angles, _ = _unpack_batch(batch)

        is_last_batch = data_iter_step == len(data_loader) - 1
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch

        current_mask_ratio = random.uniform(0.6, 0.9) if args.dynamic_mask_ratio else args.mask_ratio

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
            loss, _, _ = model(
                views,
                angles,
                gauge_angles=gauge_angles,
                mask_ratio=current_mask_ratio,
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
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

        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: iter, device: torch.device, epoch: int, args=None):
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = f'Evaluate Epoch: [{epoch}]'
    model.eval()

    val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    device_type = device.type
    autocast_enabled = device_type == 'cuda'

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views, angles, gauge_angles, _ = _unpack_batch(batch)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
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
            target_flat = views.view(B, N, -1)
            patch_mean = target_flat.mean(dim=-1, keepdim=True)
            patch_var = target_flat.var(dim=-1, keepdim=True)
            patch_std = (patch_var + 1e-6).sqrt()
            pred_flat = pred_views.view(B, N, -1)
            pred_in_global_space = (pred_flat * patch_std + patch_mean).view(B, N, C, H, W)
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

            pano_w = getattr(args, 'pano_w', 1024)
            grid_height = getattr(args, 'grid_height', int(round((N / 2) ** 0.5)))
            u_steps = 2 * grid_height

            gray_tensor = torch.full_like(ori_patches[0:1], 0.5)
            mask_expanded = mask_patches.view(N, 1, 1, 1)
            masked_input_patches = torch.where(mask_expanded > 0.5, gray_tensor, ori_patches)
            blended_patches = torch.where(mask_expanded > 0.5, rec_patches, ori_patches)

            grid_img_ori = torchvision.utils.make_grid(ori_patches, nrow=u_steps, padding=2)
            grid_img_masked = torchvision.utils.make_grid(masked_input_patches, nrow=u_steps, padding=2)
            grid_img_blend = torchvision.utils.make_grid(blended_patches, nrow=u_steps, padding=2)

            pano_ori = _unpatchify_grid(ori_patches, grid_height)
            pano_rec = _unpatchify_grid(rec_patches, grid_height)
            pano_blend = _unpatchify_grid(blended_patches, grid_height)

            grid_img_ori = _resize_to_width(grid_img_ori, pano_w)
            grid_img_masked = _resize_to_width(grid_img_masked, pano_w)
            grid_img_blend = _resize_to_width(grid_img_blend, pano_w)

            final_tensor = torch.cat([
                grid_img_ori,
                grid_img_masked,
                grid_img_blend,
                pano_rec,
                pano_blend,
                pano_ori,
            ], dim=1)

            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f'diagnostic_grid_epoch_{epoch}.png')
            torchvision.utils.save_image(final_tensor, save_path)

    psnr_value = val_psnr.compute().item()
    ssim_value = val_ssim.compute().item()

    metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.update(psnr=psnr_value, ssim=ssim_value)

    print('Averaged validation stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
