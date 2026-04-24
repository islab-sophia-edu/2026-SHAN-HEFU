import math
import sys
import os
import torch
import util.misc as misc
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import util.odi_processing as odi_helper

# =========================================================
# Metric Calculation
# =========================================================

def compute_normal_metrics(pred, target, mask):
    """
    计算 Normal 指标
    pred, target: (N, 3, H, W)
    mask: (N, 1, H, W)
    """
    pred = F.normalize(pred, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    
    valid_mask = mask.view(-1) > 0.5
    
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pred_vec = pred.permute(0, 2, 3, 1).reshape(-1, 3)[valid_mask]
    target_vec = target.permute(0, 2, 3, 1).reshape(-1, 3)[valid_mask]
    
    # 限制数值范围防止 NaN
    dot = torch.sum(pred_vec * target_vec, dim=1).clamp(-1.0, 1.0)
    angle_rad = torch.acos(dot)
    angle_deg = torch.rad2deg(angle_rad)
    
    mean_err = angle_deg.mean().item()
    median_err = angle_deg.median().item()
    rmse = torch.sqrt((angle_deg ** 2).mean()).item()
    
    p_11 = (angle_deg < 11.25).float().mean().item() * 100
    p_22 = (angle_deg < 22.5).float().mean().item() * 100
    p_30 = (angle_deg < 30.0).float().mean().item() * 100
    
    return mean_err, median_err, rmse, p_11, p_22, p_30

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
        return torch.tensor(0.0).to(pred.device)

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
    
    # 添加指标监控
    metric_logger.add_meter('mean', misc.SmoothedValue(window_size=20, fmt='{value:.2f}'))
    metric_logger.add_meter('rmse', misc.SmoothedValue(window_size=20, fmt='{value:.2f}'))
    metric_logger.add_meter('p11', misc.SmoothedValue(window_size=20, fmt='{value:.2f}'))
    metric_logger.add_meter('p30', misc.SmoothedValue(window_size=20, fmt='{value:.2f}'))
    
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    
    optimizer.zero_grad()
    num_steps_per_epoch = len(data_loader)

    for data_iter_step, (views, angles, targets, mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * num_steps_per_epoch + data_iter_step
        max_steps = args.epochs * num_steps_per_epoch
        adjust_learning_rate(optimizer, epoch, args, global_step, max_steps)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles) 
            B, N, C, H, W = logits.shape
            
            # Upsample predictions to target patch size
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            logits_upsampled = F.interpolate(logits.view(B*N, C, H, W), size=(patch_h, patch_w), mode='bilinear', align_corners=False) 
            
            targets_flat = targets.view(B*N, 3, targets.shape[-2], targets.shape[-1])
            targets_resized = F.interpolate(targets_flat, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            
            mask_flat = mask.view(B*N, 1, mask.shape[-2], mask.shape[-1])
            mask_resized = F.interpolate(mask_flat, size=(patch_h, patch_w), mode='nearest')
            
            # Loss
            loss = criterion(logits_upsampled, targets_resized, mask_resized)
            
            # Metrics (Compute on GPU)
            mean_err, median_err, rmse, p11, p22, p30 = compute_normal_metrics(
                logits_upsampled.detach().float(), 
                targets_resized.detach().float(), 
                mask_resized.detach().float()
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        # Log metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(mean=mean_err)
        metric_logger.update(rmse=rmse)
        metric_logger.update(p11=p11)
        metric_logger.update(p30=p30)
        metric_logger.update(lr=max([g["lr"] for g in optimizer.param_groups]))
    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# =========================================================
# Evaluate Function
# =========================================================

@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args, epoch=0):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    # 累加器
    total_mean = 0.0
    total_rmse = 0.0
    total_p11 = 0.0
    total_p22 = 0.0
    total_p30 = 0.0
    count = 0
    
    # 预定义的 Mean/Std 用于反归一化可视化
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

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
            mean_err, med_err, rmse, p11, p22, p30 = compute_normal_metrics(logits_upsampled, targets_resized, mask_resized)
            
            total_mean += mean_err
            total_rmse += rmse
            total_p11 += p11
            total_p22 += p22
            total_p30 += p30
            count += 1
        
        metric_logger.update(loss=loss.item())

        # ==========================================
        # GPU Visualization (First Batch Only)
        # ==========================================
        if batch_idx == 0 and misc.is_main_process():
            # 取第一个样本 (B=0)
            sample_idx = 0
            
            # 1. 准备 RGB (反归一化)
            # views: (B, N, 6, H, W), take first 3 channels
            img_tensor = views[sample_idx, :, :3, :, :] # (N, 3, H, W)
            img_tensor = img_tensor * norm_std + norm_mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 2. 准备 GT Normal (映射回 0-1)
            # targets_resized: (B*N, 3, H, W) -> view as (B, N, ...)
            gt_tensor = targets_resized.view(B, N, 3, patch_h, patch_w)[sample_idx]
            gt_tensor = (F.normalize(gt_tensor, p=2, dim=1) + 1.0) / 2.0
            gt_tensor = torch.clamp(gt_tensor, 0, 1)
            
            # 3. 准备 Pred Normal (映射回 0-1)
            pred_tensor = logits_upsampled.view(B, N, 3, patch_h, patch_w)[sample_idx]
            pred_tensor = (F.normalize(pred_tensor, p=2, dim=1) + 1.0) / 2.0
            pred_tensor = torch.clamp(pred_tensor, 0, 1)
            
            # 4. 获取 Angles (B, N, 2) -> (N, 2)
            current_angles = angles[sample_idx] # (N, 2)
            
            # 5. GPU 拼接
            # 需要计算 FOV (根据 grid_height)
            v_fov = 180.0 / args.grid_height
            h_fov = 360.0 / (2 * args.grid_height)
            
            pano_rgb = odi_helper.embed_patches_to_pano_gpu(
                patches_tensor=img_tensor,
                pano_h=args.pano_h,
                pano_w=args.pano_w,
                h_num=args.grid_height * 2,
                v_num=args.grid_height,
                h_fov_deg=h_fov,
                v_fov_deg=v_fov,
                angles=current_angles
            )
            
            pano_gt = odi_helper.embed_patches_to_pano_gpu(
                patches_tensor=gt_tensor,
                pano_h=args.pano_h,
                pano_w=args.pano_w,
                h_num=args.grid_height * 2,
                v_num=args.grid_height,
                h_fov_deg=h_fov,
                v_fov_deg=v_fov,
                angles=current_angles
            )
            
            pano_pred = odi_helper.embed_patches_to_pano_gpu(
                patches_tensor=pred_tensor,
                pano_h=args.pano_h,
                pano_w=args.pano_w,
                h_num=args.grid_height * 2,
                v_num=args.grid_height,
                h_fov_deg=h_fov,
                v_fov_deg=v_fov,
                angles=current_angles
            )
            
            # 6. 保存合并图像
            # 转换为 PIL
            def to_pil(t):
                return Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            
            pil_rgb = to_pil(pano_rgb)
            pil_gt = to_pil(pano_gt)
            pil_pred = to_pil(pano_pred)
            
            total_w = pil_rgb.width
            total_h = pil_rgb.height * 3
            combined = Image.new('RGB', (total_w, total_h))
            combined.paste(pil_rgb, (0, 0))
            combined.paste(pil_gt, (0, pil_rgb.height))
            combined.paste(pil_pred, (0, pil_rgb.height * 2))
            
            save_name = f'val_normal_epoch_{epoch:03d}.png'
            save_path = os.path.join(args.output_dir, save_name)
            combined.save(save_path)
            print(f"[Visual] Saved visualization to {save_path}")

    # Final Stats
    final_mean = total_mean / count if count > 0 else 0
    final_rmse = total_rmse / count if count > 0 else 0
    final_p11 = total_p11 / count if count > 0 else 0
    final_p22 = total_p22 / count if count > 0 else 0
    final_p30 = total_p30 / count if count > 0 else 0
    
    print(f"Val Mean: {final_mean:.2f}, RMSE: {final_rmse:.2f}, <11.25: {final_p11:.2f}%, <22.5: {final_p22:.2f}%, <30: {final_p30:.2f}%")
    
    return {
        'loss': metric_logger.loss.global_avg, 
        'mean': final_mean, 
        'rmse': final_rmse, 
        'p11': final_p11,
        'p22': final_p22,
        'p30': final_p30
    }