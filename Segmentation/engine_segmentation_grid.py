import math
import sys
import torch
import util.misc as misc
import numpy as np
from PIL import Image
import torch.nn.functional as F


def _unpack_seg_batch(batch):
    """
    Backward-compatible segmentation batch unpacking.

    PB/center-only grid dataset:
        views, angles, targets

    GCTT-compatible grid dataset:
        views, angles, gauge_angles, targets
    """
    if len(batch) == 4:
        views, angles, gauge_angles, targets = batch
    elif len(batch) == 3:
        views, angles, targets = batch
        gauge_angles = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}. Expected 3 or 4.")
    return views, angles, gauge_angles, targets


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

    valid = target != 255
    pred = pred[valid]
    target = target[valid]

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds & target_inds).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
        target_counts[cls] = target_inds.sum()

    return intersection, union, target_counts


def stitch_patches(patches_list, args):
    """
    Pure grid stitcher for diagnostics.

    This intentionally replaces the old ODI/tangent `embed_patches_to_pano` path.
    It assumes row-major patch order: row = i // (2*grid_height), col = i % (2*grid_height).
    """
    if not patches_list:
        return None

    v_steps = args.grid_height
    u_steps = 2 * args.grid_height
    first = patches_list[0]
    is_label = first.ndim == 2

    if is_label:
        patch_h, patch_w = first.shape
        out = np.zeros((v_steps * patch_h, u_steps * patch_w), dtype=first.dtype)
        for i, p in enumerate(patches_list):
            r = i // u_steps
            c = i % u_steps
            out[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w] = p
        return out

    patch_h, patch_w, channels = first.shape
    out = np.zeros((v_steps * patch_h, u_steps * patch_w, channels), dtype=first.dtype)
    for i, p in enumerate(patches_list):
        r = i // u_steps
        c = i % u_steps
        out[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w] = p
    return out


def colorize_mask(mask_tensor, palette):
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask_np).convert("P")
    img.putpalette(palette)
    return np.array(img.convert("RGB"))


def adjust_learning_rate(optimizer, epoch, config, step, max_steps):
    del epoch
    warmup_steps = config.warmup_epochs * max_steps / config.epochs
    if step < warmup_steps:
        lr = config.lr * step / max(warmup_steps, 1.0)
    else:
        loss_step = step - warmup_steps
        total_loss_steps = max(max_steps - warmup_steps, 1.0)
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * loss_step / total_loss_steps)
        )

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def _resize_logits_and_targets(logits, targets, args):
    B, N, C, H, W = logits.shape
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)

    logits_upsampled = F.interpolate(
        logits.view(B * N, C, H, W),
        size=(patch_h, patch_w),
        mode="bilinear",
        align_corners=False,
    )

    targets_resized = F.interpolate(
        targets.view(B * N, 1, targets.shape[-2], targets.shape[-1]).float(),
        size=(patch_h, patch_w),
        mode="nearest",
    ).long().squeeze(1)

    return logits_upsampled, targets_resized


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", misc.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()
    num_steps_per_epoch = len(data_loader)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        views, angles, gauge_angles, targets = _unpack_seg_batch(batch)

        global_step = epoch * num_steps_per_epoch + data_iter_step
        max_steps = args.epochs * num_steps_per_epoch
        adjust_learning_rate(optimizer, epoch, args, global_step, max_steps)

        is_last_batch = data_iter_step == len(data_loader) - 1
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(
                views,
                angles,
                gauge_angles=gauge_angles,
            )
            logits_upsampled, targets_resized = _resize_logits_and_targets(logits, targets, args)
            loss = criterion(logits_upsampled, targets_resized)

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

        metric_logger.update(loss=loss_value)

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args, epoch=0):
    del epoch
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    total_inter = torch.zeros(args.nb_classes, device=device)
    total_union = torch.zeros(args.nb_classes, device=device)
    total_target = torch.zeros(args.nb_classes, device=device)

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        del batch_idx
        views, angles, gauge_angles, targets = _unpack_seg_batch(batch)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(
                views,
                angles,
                gauge_angles=gauge_angles,
            )
            logits_upsampled, targets_resized = _resize_logits_and_targets(logits, targets, args)
            loss = criterion(logits_upsampled, targets_resized)

            pred_labels = logits_upsampled.argmax(dim=1)
            inter, union, target_cnt = compute_metrics(pred_labels, targets_resized, args.nb_classes)
            total_inter += inter
            total_union += union
            total_target += target_cnt

        metric_logger.update(loss=loss.item())

    iou_per_class = total_inter / (total_union + 1e-6)
    miou = iou_per_class.mean().item()

    acc_per_class = total_inter / (total_target + 1e-6)
    macc = acc_per_class.mean().item()

    print(f"Validation mIoU: {miou:.4f}, mAcc: {macc:.4f}")

    return {
        "loss": metric_logger.loss.global_avg,
        "miou": miou,
        "macc": macc,
    }
