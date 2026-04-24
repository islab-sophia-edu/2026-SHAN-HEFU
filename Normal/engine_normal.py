import math
import sys
import os
import torch
import util.misc as misc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Metric Calculation (Updated with P5, P7.5, etc.)
# =========================================================

def compute_normal_metrics(pred, target, mask):
    """
    计算 Normal 指标。
    Thresholds: 5, 7.5, 11.25, 22.5, 30 degrees.
    """
    # 归一化 (Safety First)
    pred = F.normalize(pred, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    
    # 展平 Mask，只计算有效像素
    valid_mask = mask.view(-1) > 0.5
    count = valid_mask.sum().item()
    
    # 边界情况处理：如果该 Batch 没有有效像素
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 提取有效像素向量 (M, 3)
    # permute(0,2,3,1) -> (B, H, W, C) -> reshape(-1, 3) -> mask indexing
    pred_vec = pred.permute(0, 2, 3, 1).reshape(-1, 3)[valid_mask]
    target_vec = target.permute(0, 2, 3, 1).reshape(-1, 3)[valid_mask]
    
    # 计算点积和角度
    dot = torch.sum(pred_vec * target_vec, dim=1).clamp(-1.0, 1.0)
    angle_rad = torch.acos(dot)
    angle_deg = torch.rad2deg(angle_rad)
    
    # --- Core Metrics ---
    mean_err = angle_deg.mean().item()
    rmse = torch.sqrt((angle_deg ** 2).mean()).item()
    
    # --- Percentage Metrics (Accurate Thresholds) ---
    # P5 (5 deg)
    p_5 = (angle_deg < 5.0).float().mean().item() * 100
    # P7 (7.5 deg)
    p_7 = (angle_deg < 7.5).float().mean().item() * 100
    # P11 (11.25 deg)
    p_11 = (angle_deg < 11.25).float().mean().item() * 100
    # P22 (22.5 deg)
    p_22 = (angle_deg < 22.5).float().mean().item() * 100
    # P30 (30 deg)
    p_30 = (angle_deg < 30.0).float().mean().item() * 100
    
    return mean_err, rmse, p_5, p_7, p_11, p_22, p_30, count

class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, mask):
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        cosine_sim = torch.sum(pred * target, dim=1, keepdim=True)
        loss_map = 1.0 - cosine_sim
        
        valid_mask = mask > 0.5
        if valid_mask.sum() > 0:
            return loss_map[valid_mask].mean()
        # 保持梯度图完整性
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

# =========================================================
# Helper Functions
# =========================================================

def adjust_learning_rate(optimizer, epoch, config, step, max_steps):
    if step < config.warmup_epochs * max_steps / config.epochs:
        lr = config.lr * step / (config.warmup_epochs * max_steps / config.epochs) 
    else:
        loss_step = step - config.warmup_epochs * max_steps / config.epochs
        total_loss_steps = max_steps - config.warmup_epochs * max_steps / config.epochs
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * loss_step / total_loss_steps))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# =========================================================
# Train Function
# =========================================================

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Smoothed meters for real-time logging during training
    for k in ['mean', 'rmse', 'p5', 'p7', 'p11', 'p22', 'p30']:
        metric_logger.add_meter(k, misc.SmoothedValue(window_size=20, fmt='{value:.2f}'))
    
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    optimizer.zero_grad()
    
    num_steps_per_epoch = len(data_loader)
    total_valid_pixels = 0
    accum_metrics = {
        'mean': 0.0, 'rmse': 0.0, 
        'p5': 0.0, 'p7': 0.0, 
        'p11': 0.0, 'p22': 0.0, 'p30': 0.0
    }

    for data_iter_step, (views, angles, targets, mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * num_steps_per_epoch + data_iter_step
        
        # LR Schedule
        if hasattr(args, 'warmup_epochs'): # 增加健壮性检查
             adjust_learning_rate(optimizer, epoch, args, global_step, args.epochs * num_steps_per_epoch)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)
            
            # Resize Logic
            B, N, C = logits.shape[:3]
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            logits_up = F.interpolate(logits.view(B*N, C, logits.shape[-2], logits.shape[-1]), size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            targets_up = F.interpolate(targets.view(B*N, 3, targets.shape[-2], targets.shape[-1]), size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            mask_up = F.interpolate(mask.view(B*N, 1, mask.shape[-2], mask.shape[-1]), size=(patch_h, patch_w), mode='nearest')
            
            loss = criterion(logits_up, targets_up, mask_up)
            
            # Metrics (Detach to prevent graph retention)
            mean_err, rmse, p5, p7, p11, p22, p30, batch_count = compute_normal_metrics(
                logits_up.detach().float(), 
                targets_up.detach().float(), 
                mask_up.detach().float()
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        # Update Smoothed Logs (For visual progress)
        metric_logger.update(loss=loss_value)
        metric_logger.update(mean=mean_err)
        metric_logger.update(rmse=rmse)
        metric_logger.update(p5=p5)
        metric_logger.update(p7=p7)
        metric_logger.update(p11=p11)
        metric_logger.update(p22=p22)
        metric_logger.update(p30=p30)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Accumulate Exact Stats (For final report)
        if batch_count > 0:
            accum_metrics['mean'] += mean_err * batch_count
            accum_metrics['rmse'] += rmse * batch_count
            accum_metrics['p5']   += p5 * batch_count
            accum_metrics['p7']   += p7 * batch_count
            accum_metrics['p11']  += p11 * batch_count
            accum_metrics['p22']  += p22 * batch_count
            accum_metrics['p30']  += p30 * batch_count
            total_valid_pixels += batch_count
    
    # Final Epoch Stats
    final_stats = {k: (v / total_valid_pixels if total_valid_pixels > 0 else 0.0) for k, v in accum_metrics.items()}

    print(f"Epoch [{epoch}] Final Stats:")
    print(f"  Mean: {final_stats['mean']:.2f}, RMSE: {final_stats['rmse']:.2f}")
    print(f"  P5:   {final_stats['p5']:.2f}%, P7.5: {final_stats['p7']:.2f}%")
    print(f"  P11.25: {final_stats['p11']:.2f}%, P22.5: {final_stats['p22']:.2f}%")

    return {
        'loss': metric_logger.meters['loss'].global_avg,
        'lr': metric_logger.meters['lr'].global_avg,
        **final_stats
    }
# =========================================================
# Evaluate Function
# =========================================================

@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args, epoch=0):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    # 累加器 (使用加权平均)
    total_samples = 0
    accum_metrics = {
        'mean': 0.0, 'rmse': 0.0, 
        'p5': 0.0, 'p7': 0.0, 
        'p11': 0.0, 'p22': 0.0, 'p30': 0.0
    }

    for batch_idx, (views, angles, targets, mask) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)
            B, N, C = logits.shape[:3]
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            # Upsample
            logits_upsampled = F.interpolate(logits.view(B*N, C, logits.shape[-2], logits.shape[-1]), size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            
            targets_flat = targets.view(B*N, 3, targets.shape[-2], targets.shape[-1])
            targets_resized = F.interpolate(targets_flat, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            
            mask_flat = mask.view(B*N, 1, mask.shape[-2], mask.shape[-1])
            mask_resized = F.interpolate(mask_flat, size=(patch_h, patch_w), mode='nearest')

            loss = criterion(logits_upsampled, targets_resized, mask_resized)
            
            # Metrics
            mean_err, rmse, p5, p7, p11, p22, p30, batch_count = compute_normal_metrics(
                logits_upsampled, targets_resized, mask_resized
            )
            
            # Accumulate
            if batch_count > 0:
                accum_metrics['mean'] += mean_err * batch_count
                accum_metrics['rmse'] += rmse * batch_count
                accum_metrics['p5']   += p5 * batch_count
                accum_metrics['p7']   += p7 * batch_count
                accum_metrics['p11']  += p11 * batch_count
                accum_metrics['p22']  += p22 * batch_count
                accum_metrics['p30']  += p30 * batch_count
                total_samples += batch_count
        
        metric_logger.update(loss=loss.item())

    # Final Calculation
    final_stats = {k: (v / total_samples if total_samples > 0 else 0.0) for k, v in accum_metrics.items()}
    
    print(f"Val Results:")
    print(f"  Mean: {final_stats['mean']:.2f}, RMSE: {final_stats['rmse']:.2f}")
    print(f"  P5:   {final_stats['p5']:.2f}%, P7: {final_stats['p7']:.2f}%")
    print(f"  P11:  {final_stats['p11']:.2f}%, P22:  {final_stats['p22']:.2f}%, P30: {final_stats['p30']:.2f}%")
    
    return {
        'loss': metric_logger.loss.global_avg, 
        **final_stats
    }