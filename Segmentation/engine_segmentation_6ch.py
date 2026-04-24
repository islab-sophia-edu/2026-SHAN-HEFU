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

def compute_metrics(pred, target, num_classes):
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    target_counts = torch.zeros(num_classes, device=pred.device)

    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds & target_inds).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
        target_counts[cls] = target_inds.sum()
        
    return intersection, union, target_counts

def stitch_patches(patches_list, args):
    """
    patches_list: List of np.array, each (H, W) or (H, W, 3)
    """
    first_patch = patches_list[0]
    
    # 判断是否为彩色图 (H, W, 3)
    is_color = (first_patch.ndim == 3 and first_patch.shape[2] == 3)
    
    # 如果是分割结果（已经上色），它已经是 RGB 了，可以直接拼
    # 如果是标签 Label (H, W)，需要 fake RGB 才能拼
    
    if not is_color:
        # 转为伪三通道 (复制3次) 以利用 odi 代码
        patches_rgb_fake = [np.stack([p, p, p], axis=-1) for p in patches_list]
        patches_to_stitch = patches_rgb_fake
    else:
        patches_to_stitch = patches_list

    pano_rgb = odi_helper.embed_patches_to_pano(
        patches_np_list=patches_to_stitch,
        pano_size=(args.pano_h, args.pano_w),
        h_num=args.grid_height * 2,
        v_num=args.grid_height,
        h_fov_deg=360.0 / (args.grid_height * 2),
        v_fov_deg=180.0 / args.grid_height
    )
    
    # 如果原本是单通道标签，拼完后取回单通道
    if not is_color:
        return pano_rgb[:, :, 0].astype(np.int64)
    else:
        return pano_rgb

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
            logits = model(views, angles) 
            B, N, C, H, W = logits.shape
            
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, H, W), 
                size=(patch_h, patch_w), 
                mode='bilinear', 
                align_corners=False
            ) 
            
            targets_resized = F.interpolate(
                targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]).float(),
                size=(patch_h, patch_w),
                mode='nearest'
            ).long().squeeze(1) 
            
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
            logits = model(views, angles) 
            B, N, C = logits.shape[:3]
            
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, logits.shape[-2], logits.shape[-1]),
                size=(patch_h, patch_w),
                mode='bilinear',
                align_corners=False
            )
            
            targets_flat = targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]).float()
            targets_resized = F.interpolate(
                targets_flat,
                size=(patch_h, patch_w), 
                mode='nearest'
            ).long().squeeze(1) 

            loss = criterion(logits_upsampled, targets_resized)

            pred_labels = logits_upsampled.argmax(dim=1) 
            
            inter, union, target_cnt = compute_metrics(pred_labels, targets_resized, args.nb_classes)
            total_inter += inter
            total_union += union
            total_target += target_cnt
        
        metric_logger.update(loss=loss.item())
        '''
        # [修复] 仅使用 RGB 通道进行可视化
        if batch_idx == 0 and misc.is_main_process():
            # print("Stitching visualization for the first sample... (This may take a moment)")
            sample_idx = 0 
            
            # 1. 拆分 6 通道，只取前 3 通道 (RGB)
            # views: (B, N, 6, H, W)
            views_rgb = views[:, :, :3, :, :] 
            
            # 2. 反归一化
            raw_rgb = views_rgb[sample_idx] * std[0] + mean[0]
            raw_rgb = torch.clamp(raw_rgb, 0, 1) # (N, 3, 52, 52)
            
            rgb_patches = []
            gt_patches = []
            pred_patches = []
            
            pred_labels_vis = pred_labels.view(B, N, patch_h, patch_w)[sample_idx] 
            targets_vis = targets_resized.view(B, N, patch_h, patch_w)[sample_idx] 
            
            palette = get_color_palette(args.nb_classes)

            for i in range(N):
                # RGB
                p_rgb = raw_rgb[i].permute(1, 2, 0).cpu().float().numpy()
                rgb_patches.append(p_rgb)
                
                # GT
                p_gt = colorize_mask(targets_vis[i], palette)
                gt_patches.append(p_gt.astype(np.float32) / 255.0)
                
                # Pred
                p_pred = colorize_mask(pred_labels_vis[i], palette)
                pred_patches.append(p_pred.astype(np.float32) / 255.0)

            pano_rgb = stitch_patches(rgb_patches, args)
            pano_gt = stitch_patches(gt_patches, args)
            pano_pred = stitch_patches(pred_patches, args)
            
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
            combined.save(save_path)'''
            # print(f"Saved visualization to {save_path}")

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