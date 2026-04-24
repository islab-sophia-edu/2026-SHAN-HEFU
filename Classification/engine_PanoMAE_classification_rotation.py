import math
import sys
import numpy as np
import torch
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, max_norm=0, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('acc1', misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
    
    header = 'Epoch: [{}]'.format(epoch)

    # 获取 MixUp / CutMix 超参
    mixup_alpha = getattr(args, 'mixup', 0.8)
    cutmix_alpha = getattr(args, 'cutmix', 1.0)
    mixup_prob = getattr(args, 'mixup_prob', 1.0)

    for data_iter_step, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if data_iter_step % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ===================================================================
        # [核心突破] 专为全景 3dSPE 设计的 Token-level MixUp & CutMix
        # ===================================================================
        do_mixup = (mixup_alpha > 0 or cutmix_alpha > 0) and (np.random.rand() < mixup_prob)
        if do_mixup:
            B, N, C, H, W = views.shape
            # 随机决定当前 Batch 用 CutMix 还是 MixUp (50/50 概率)
            use_cutmix = (cutmix_alpha > 0) and (np.random.rand() < 0.5 or mixup_alpha == 0)
            
            index = torch.randperm(B).to(device)
            targets_a = targets
            targets_b = targets[index]
            
            if use_cutmix:
                # --- Token-level CutMix (PatchMix) ---
                lam_dist = np.random.beta(cutmix_alpha, cutmix_alpha)
                # 为每个 Patch 生成替换 Mask
                mask = torch.rand((B, N), device=device) < lam_dist
                lam = mask.float().mean().item() # 获取真实的 lambda 比例
                
                # 同步替换视觉 Patch 和 它的经纬度 Angle！(完美保护 3dSPE)
                mask_views = mask.view(B, N, 1, 1, 1).expand_as(views)
                views = torch.where(mask_views, views, views[index])
                
                mask_angles = mask.view(B, N, 1).expand_as(angles)
                angles = torch.where(mask_angles, angles, angles[index])
            else:
                # --- Dominant-A MixUp ---
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                if lam < 0.5: 
                    lam = 1.0 - lam # 强制 A 作为主导背景，B 作为纹理叠加
                views = lam * views + (1 - lam) * views[index]
                # 角度坐标系直接保留主导图 A 的 angles
        
        # ===================================================================
        
        with torch.cuda.amp.autocast():
            outputs = model(views, angles)
            if do_mixup:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        # 评估训练准确率 (若使用了混合，以占主导地位的 label 作为衡量标准)
        eval_targets = targets_a if (do_mixup and lam >= 0.5) else (targets_b if do_mixup else targets)
        acc1, acc5 = accuracy(outputs, eval_targets, topk=(1, 5))
        
        batch_size = views.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()

    for views, angles, target in metric_logger.log_every(data_loader, 10, 'Test:'):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(views, angles)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = views.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}