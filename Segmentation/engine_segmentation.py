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

def get_color_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

# [修改] 改名为 compute_metrics 并多返回一个 target_counts
def compute_metrics(pred, target, num_classes):
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    target_counts = torch.zeros(num_classes, device=pred.device) # [新增] 记录真实标签数量

    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds & target_inds).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
        target_counts[cls] = target_inds.sum() # [新增]
        
    return intersection, union, target_counts

def stitch_patches(patches_list, args):
    """
    patches_list: List of np.array, each (H, W) for segmentation labels
    """
    # [新增] 检测是否为单通道分割标签
    first_patch = patches_list[0]
    is_label = (first_patch.ndim == 2)  # 单通道
    
    if is_label:
        # 转为伪三通道 (复制3次)
        patches_rgb_fake = [np.stack([p, p, p], axis=-1) for p in patches_list]
        
        pano_rgb = odi_helper.embed_patches_to_pano(
            patches_np_list=patches_rgb_fake,
            pano_size=(args.pano_h, args.pano_w),
            h_num=args.grid_height * 2,
            v_num=args.grid_height,
            h_fov_deg=360.0 / (args.grid_height * 2),
            v_fov_deg=180.0 / args.grid_height
        )
        
        # 提取第一个通道作为标签
        return pano_rgb[:, :, 0].astype(np.int64)
    else:
        # 原有的RGB处理逻辑
        return odi_helper.embed_patches_to_pano(
            patches_np_list=patches_list,
            pano_size=(args.pano_h, args.pano_w),
            h_num=args.grid_height * 2,
            v_num=args.grid_height,
            h_fov_deg=360.0 / (args.grid_height * 2),
            v_fov_deg=180.0 / args.grid_height
        )
        
def colorize_mask(mask_tensor, palette):
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask_np).convert('P')
    img.putpalette(palette)
    return np.array(img.convert('RGB'))

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

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'
    
    optimizer.zero_grad()
    num_steps_per_epoch = len(data_loader)

    for data_iter_step, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        global_step = epoch * num_steps_per_epoch + data_iter_step
        max_steps = args.epochs * num_steps_per_epoch
        adjust_learning_rate(optimizer, epoch, args, global_step, max_steps)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)  # (B, N, num_classes, 8, 8)
            B, N, C, H, W = logits.shape
            
            # 上采样到patch目标尺寸
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, H, W), 
                size=(patch_h, patch_w), 
                mode='bilinear', 
                align_corners=False
            )  # (B*N, C, patch_h, patch_w)
            
            targets_resized = F.interpolate(
                targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]).float(),
                size=(patch_h, patch_w),
                mode='nearest'
            ).long().squeeze(1)  # (B*N, patch_h, patch_w)
            
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
    
    total_inter = torch.zeros(args.nb_classes, device=device)
    total_union = torch.zeros(args.nb_classes, device=device)
    total_target = torch.zeros(args.nb_classes, device=device)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    for batch_idx, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)  # (B, N, num_classes, 8, 8)
            B, N, C = logits.shape[:3]
            
            # 1. 上采样 Logits 到 Patch 尺寸 (例如 52x52)
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            # (B*N, C, H, W)
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, logits.shape[-2], logits.shape[-1]),
                size=(patch_h, patch_w),
                mode='bilinear',
                align_corners=False
            )
            
            # 2. 处理 Targets (Resize 如果需要，或者直接用)
            # 注意: targets 已经是 (B, N, patch_h, patch_w) 或者是 (B, N, 52, 52)
            # 为了保险，我们强制 resize 到和 logits 一样
            targets_flat = targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]).float()
            targets_resized = F.interpolate(
                targets_flat,
                size=(patch_h, patch_w), 
                mode='nearest'
            ).long().squeeze(1) # (B*N, patch_h, patch_w)

            # 3. 计算 Loss (在 Patch 级别)
            loss = criterion(logits_upsampled, targets_resized)

            # 4. 计算 Metrics (直接在 Patch 级别计算，大幅加速)
            # 以前这里卡住了，因为调用了 stitch_patches
            pred_labels = logits_upsampled.argmax(dim=1) # (B*N, patch_h, patch_w)
            
            inter, union, target_cnt = compute_metrics(pred_labels, targets_resized, args.nb_classes)
            total_inter += inter
            total_union += union
            total_target += target_cnt
        
        metric_logger.update(loss=loss.item())

        # 5. 可视化 (仅在第一个 Batch 的第一个样本执行拼接，避免卡顿)
        """if batch_idx == 0 and misc.is_main_process():
            print("Stitching visualization for the first sample... (This may take a moment)")
            sample_idx = 0 # 只取 Batch 中的第 0 张
            
            # 还原 RGB
            raw_rgb = views[sample_idx] * std[0] + mean[0]
            raw_rgb = torch.clamp(raw_rgb, 0, 1) # (N, 3, 52, 52)
            
            rgb_patches = []
            gt_patches = []
            pred_patches = []
            
            # 准备 Patch 列表
            # 注意：这里需要先把 Tensor 转回 numpy
            pred_labels_vis = pred_labels.view(B, N, patch_h, patch_w)[sample_idx] # (N, H, W)
            targets_vis = targets_resized.view(B, N, patch_h, patch_w)[sample_idx] # (N, H, W)
            
            palette = get_color_palette(args.nb_classes)

            for i in range(N):
                # RGB
                p_rgb = raw_rgb[i].permute(1, 2, 0).cpu().float().numpy()
                rgb_patches.append(p_rgb)
                
                # GT (Colorized)
                p_gt = colorize_mask(targets_vis[i], palette)
                gt_patches.append(p_gt.astype(np.float32) / 255.0)
                
                # Pred (Colorized)
                p_pred = colorize_mask(pred_labels_vis[i], palette)
                pred_patches.append(p_pred.astype(np.float32) / 255.0)

            # --- 耗时的拼接操作仅在此处执行一次 ---
            pano_rgb = stitch_patches(rgb_patches, args)
            
            # 注意：stitch_patches 对于 mask 需要特殊处理，它期望的是 List[H,W] 或者 List[H,W,3]
            # 我们这里传入的是已经 colorize 过的 List[H,W,3] (fake RGB)，所以 stitch_patches 会直接拼
            pano_gt = stitch_patches(gt_patches, args)
            pano_pred = stitch_patches(pred_patches, args)
            
            # 保存图片
            img_rgb = Image.fromarray((pano_rgb * 255).astype(np.uint8))
            img_gt = Image.fromarray((pano_gt * 255).astype(np.uint8))
            img_pred = Image.fromarray((pano_pred * 255).astype(np.uint8))
            
            total_w = img_rgb.width
            total_h = img_rgb.height * 3
            combined = Image.new('RGB', (total_w, total_h))
            combined.paste(img_rgb, (0, 0))
            combined.paste(img_gt, (0, img_rgb.height))
            combined.paste(img_pred, (0, img_rgb.height * 2))
            
            save_name = f'val_compare_epoch_{epoch:03d}.png'
            save_path = os.path.join(args.output_dir, save_name)
            combined.save(save_path)
            print(f"Saved visualization to {save_path}")"""

    # 计算 mIoU
    iou_per_class = total_inter / (total_union + 1e-6)
    miou = iou_per_class.mean().item()
    
    # 计算 mAcc
    acc_per_class = total_inter / (total_target + 1e-6)
    macc = acc_per_class.mean().item()
    
    print(f"Validation mIoU: {miou:.4f}, mAcc: {macc:.4f}")
    
    return {
        'loss': metric_logger.loss.global_avg, 
        'miou': miou, 
        'macc': macc
    }