import math
import sys
import os
import torch
import util.misc as misc
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch

class DepthLoss(nn.Module):
    def __init__(self, w_silog=1.0, w_grad=0.5, w_berhu=0.5):
        super().__init__()
        self.w_silog = w_silog
        self.w_grad  = w_grad
        self.w_berhu = w_berhu
        self.alpha = 0.85

    def forward(self, pred, target):
        pred_f   = pred.float()
        target_f = target.float()

        mask = target_f > 0.01
        if mask.sum() < 10:
            return (pred_f * 0.0).sum()

        pred_valid   = pred_f[mask].clamp(min=1e-3)
        target_valid = target_f[mask].clamp(min=1e-3)

        # 1. SILog Loss
        log_diff = torch.log(pred_valid) - torch.log(target_valid)
        variance = (log_diff ** 2).mean() - self.alpha * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(variance.clamp(min=0.0) + 1e-6) * 10.0

        # 2. BerHu Loss
        diff_abs = torch.abs(pred_valid - target_valid)
        c = 0.2 * diff_abs.max().detach()
        berhu_loss = torch.where(
            diff_abs <= c,
            diff_abs,
            (diff_abs ** 2 + c ** 2) / (2 * c)
        ).mean()

        # 3. Gradient Loss（对数空间）
        log_pred   = torch.log(pred_f.clamp(min=1e-3))
        log_target = torch.log(target_f.clamp(min=1e-3))

        grad_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        diff_x = (log_pred[:, :, :, 1:] - log_pred[:, :, :, :-1]) - \
                 (log_target[:, :, :, 1:] - log_target[:, :, :, :-1])
        diff_y = (log_pred[:, :, 1:, :] - log_pred[:, :, :-1, :]) - \
                 (log_target[:, :, 1:, :] - log_target[:, :, :-1, :])

        mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        if mask_x.sum() > 0:
            grad_loss += (diff_x[mask_x] ** 2).mean()
        if mask_y.sum() > 0:
            grad_loss += (diff_y[mask_y] ** 2).mean()
        grad_loss = torch.sqrt(grad_loss.clamp(min=0) + 1e-6)

        total = self.w_silog * silog_loss + \
                self.w_berhu * berhu_loss + \
                self.w_grad  * grad_loss
        return total

    
def compute_depth_metrics(pred_all, target_all):
    """
    pred_all, target_all: (N,) 已拼接的全量 tensor，单位为米
    在整个验证集上统一计算，避免 batch 平均导致 RMSE 失真
    """
    pred_all   = pred_all.float()
    target_all = target_all.float()

    mask = target_all > 1e-3
    pred   = pred_all[mask].clamp(min=1e-4)
    target = target_all[mask]

    if pred.numel() == 0:
        return 0.0, 0.0, 0.0

    thresh   = torch.max(target / pred, pred / target)
    delta1   = (thresh < 1.25).float().mean()

    diff     = pred - target
    mae      = diff.abs().mean()
    rmse     = torch.sqrt((diff ** 2).mean())
    abs_rel  = (diff.abs() / target).mean()

    return mae.item(), rmse.item(), abs_rel.item(), delta1.item()


def adjust_learning_rate(optimizer, epoch, config, step, max_steps):
    if step < config.warmup_epochs * max_steps / config.epochs:
        lr = config.lr * step / (config.warmup_epochs * max_steps / config.epochs)
    else:
        loss_step       = step - config.warmup_epochs * max_steps / config.epochs
        total_loss_steps = max_steps - config.warmup_epochs * max_steps / config.epochs
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * loss_step / total_loss_steps))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',     misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'

    optimizer.zero_grad()
    num_steps_per_epoch = len(data_loader)

    for data_iter_step, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        global_step = epoch * num_steps_per_epoch + data_iter_step
        max_steps   = args.epochs * num_steps_per_epoch
        adjust_learning_rate(optimizer, epoch, args, global_step, max_steps)

        views   = views.to(device,   non_blocking=True)
        angles  = angles.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)
            B, N, C, H, W = logits.shape

            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)

            logits_upsampled = F.interpolate(
                logits.view(B * N, C, H, W),
                size=(patch_h, patch_w),
                mode='bilinear',
                align_corners=False
            )

            targets_flat    = targets.view(B * N, 1, targets.shape[-2], targets.shape[-1])
            targets_resized = F.interpolate(
                targets_flat,
                size=(patch_h, patch_w),
                mode='nearest'
            )

            # ✅ 直接使用传入的 criterion，不在循环内重建
            loss = criterion(logits_upsampled, targets_resized)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % args.accum_iter == 0)

        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args, epoch=0):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    MAX_DEPTH_METERS = 10.0

    # ✅ 收集全量预测与真值，在验证集结束后统一计算指标
    all_pred   = []
    all_target = []

    for batch_idx, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views   = views.to(device,   non_blocking=True)
        angles  = angles.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)
            B, N, C = logits.shape[:3]

            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)

            logits_upsampled = F.interpolate(
                logits.view(B * N, C, logits.shape[-2], logits.shape[-1]),
                size=(patch_h, patch_w),
                mode='bilinear',
                align_corners=False
            )

            targets_flat    = targets.view(B * N, 1, targets.shape[-2], targets.shape[-1])
            targets_resized = F.interpolate(
                targets_flat,
                size=(patch_h, patch_w),
                mode='nearest'
            )

            loss = criterion(logits_upsampled, targets_resized)

            # 反归一化到米，展平后收集
            pred_meters   = logits_upsampled.float().clamp(0, 1) * MAX_DEPTH_METERS
            target_meters = targets_resized.float() * MAX_DEPTH_METERS

            valid = (target_meters > 0) & (target_meters <= MAX_DEPTH_METERS)
            all_pred.append(pred_meters[valid].cpu())
            all_target.append(target_meters[valid].cpu())

        metric_logger.update(loss=loss.item())

    all_pred   = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    mae, rmse, abs_rel, delta1 = compute_depth_metrics(all_pred, all_target)

    print(f"Validation RMSE: {rmse:.4f}m | AbsRel: {abs_rel:.4f} | Delta1: {delta1:.4f}")

    return {
        'loss':    metric_logger.loss.global_avg,
        'mae':     mae,
        'rmse':    rmse,
        'abs_rel': abs_rel,
        'delta1':  delta1,
    }