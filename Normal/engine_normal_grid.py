import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import util.misc as misc


def _unpack_normal_batch(batch):
    if len(batch) == 5:
        views, angles, gauge_angles, targets, valid_mask = batch
    elif len(batch) == 4:
        views, angles, targets, valid_mask = batch
        gauge_angles = None
    elif len(batch) == 3:
        views, angles, targets = batch
        gauge_angles = None
        valid_mask = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}. Expected 3, 4, or 5.")
    return views, angles, gauge_angles, targets, valid_mask


class CosineNormalLoss(nn.Module):
    def __init__(self, w_cos=1.0, w_l1=0.0, bidirectional=False):
        super().__init__()
        self.w_cos = float(w_cos)
        self.w_l1 = float(w_l1)
        self.bidirectional = bool(bidirectional)

    def forward(self, pred, target, valid_mask=None):
        pred = F.normalize(pred.float(), p=2, dim=1, eps=1e-6)
        target = F.normalize(target.float(), p=2, dim=1, eps=1e-6)
        dot = (pred * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        if self.bidirectional:
            dot_for_loss = dot.abs()
            l1_term = torch.minimum(
                (pred - target).abs().mean(dim=1, keepdim=True),
                (pred + target).abs().mean(dim=1, keepdim=True),
            )
        else:
            dot_for_loss = dot
            l1_term = (pred - target).abs().mean(dim=1, keepdim=True)
        loss_map = self.w_cos * (1.0 - dot_for_loss)
        if self.w_l1 > 0:
            loss_map = loss_map + self.w_l1 * l1_term
        if valid_mask is not None:
            mask = valid_mask.float() > 0.5
        else:
            mask = target.norm(dim=1, keepdim=True) > 0.1
        if mask.sum() < 10:
            return (pred * 0.0).sum()
        return loss_map[mask].mean()


def compute_normal_metrics(pred, target, valid_mask=None, bidirectional=False):
    pred = F.normalize(pred.float(), p=2, dim=1, eps=1e-6)
    target = F.normalize(target.float(), p=2, dim=1, eps=1e-6)
    if valid_mask is not None:
        mask = valid_mask.view(-1) > 0.5
    else:
        mask = target.norm(dim=1, keepdim=True).view(-1) > 0.1
    count = mask.sum().item()
    if count == 0:
        return {"mean": 0.0, "median": 0.0, "rmse": 0.0, "p5": 0.0, "p7": 0.0, "p11": 0.0, "p22": 0.0, "p30": 0.0, "count": 0}
    pred_vec = pred.permute(0, 2, 3, 1).reshape(-1, 3)[mask]
    target_vec = target.permute(0, 2, 3, 1).reshape(-1, 3)[mask]
    dot = (pred_vec * target_vec).sum(dim=1).clamp(-1.0, 1.0)
    if bidirectional:
        dot = dot.abs()
    angle = torch.rad2deg(torch.acos(dot))
    return {
        "mean": angle.mean().item(),
        "median": angle.median().item(),
        "rmse": torch.sqrt((angle ** 2).mean()).item(),
        "p5": (angle < 5.0).float().mean().item() * 100.0,
        "p7": (angle < 7.5).float().mean().item() * 100.0,
        "p11": (angle < 11.25).float().mean().item() * 100.0,
        "p22": (angle < 22.5).float().mean().item() * 100.0,
        "p30": (angle < 30.0).float().mean().item() * 100.0,
        "count": count,
    }


def adjust_learning_rate(optimizer, epoch, config, step, max_steps):
    del epoch
    warmup_steps = config.warmup_epochs * max_steps / config.epochs
    if step < warmup_steps:
        lr = config.lr * step / max(warmup_steps, 1.0)
    else:
        loss_step = step - warmup_steps
        total_loss_steps = max(max_steps - warmup_steps, 1.0)
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * (1.0 + math.cos(math.pi * loss_step / total_loss_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr


def _resize_normal_pred_and_targets(logits, targets, args, valid_mask=None):
    """Resize model outputs and normal targets to the ERP-grid patch size."""
    B, N, C, H, W = logits.shape
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)

    pred = F.interpolate(logits.view(B * N, C, H, W), size=(patch_h, patch_w), mode="bilinear", align_corners=False)
    pred = F.normalize(pred, p=2, dim=1, eps=1e-6)

    target_raw = targets.view(B * N, 3, targets.shape[-2], targets.shape[-1]).float()
    valid = None
    if valid_mask is not None:
        valid_raw = valid_mask.view(B * N, 1, valid_mask.shape[-2], valid_mask.shape[-1]).float()
        valid_raw = (valid_raw > 0.5).float()
        target_num = F.interpolate(target_raw * valid_raw, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
        target_den = F.interpolate(valid_raw, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
        target = target_num / target_den.clamp_min(1e-6)
        target = F.normalize(target, p=2, dim=1, eps=1e-6)
        valid_thr = float(getattr(args, "normal_valid_resize_threshold", 0.5))
        valid = (target_den >= valid_thr).float()
        target = target * valid
    else:
        target = F.interpolate(target_raw, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
        target = F.normalize(target, p=2, dim=1, eps=1e-6)
    return pred, target, valid


def _get_model_aux_outputs(model):
    module = model.module if hasattr(model, "module") else model
    if hasattr(module, "get_aux_outputs"):
        return module.get_aux_outputs()
    return []


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", misc.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = f"Epoch: [{epoch}]"
    optimizer.zero_grad()
    num_steps_per_epoch = len(data_loader)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        views, angles, gauge_angles, targets, valid_mask = _unpack_normal_batch(batch)
        global_step = epoch * num_steps_per_epoch + data_iter_step
        adjust_learning_rate(optimizer, epoch, args, global_step, args.epochs * num_steps_per_epoch)
        is_last_batch = data_iter_step == len(data_loader) - 1
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(views, angles, gauge_angles=gauge_angles)
            pred, target, valid = _resize_normal_pred_and_targets(logits, targets, args, valid_mask)
            loss_main = criterion(pred, target, valid)
            aux_loss = pred.new_tensor(0.0)
            w_aux = float(getattr(args, "w_aux_normal", 0.0))
            if w_aux > 0.0:
                for aux_logits in _get_model_aux_outputs(model):
                    aux_pred, aux_target, aux_valid = _resize_normal_pred_and_targets(aux_logits, targets, args, valid_mask)
                    aux_loss = aux_loss + criterion(aux_pred, aux_target, aux_valid)
            loss = loss_main + w_aux * aux_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss = loss / args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), update_grad=is_update_step)
        if is_update_step:
            optimizer.zero_grad()
        metric_logger.update(loss=loss_value)
        metric_logger.update(aux_loss=float(aux_loss.detach().item()))
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _normal_to_rgb(normal, valid_mask=None):
    rgb = (normal.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    if valid_mask is not None:
        rgb = rgb * (valid_mask.float() > 0.5).float()
    return rgb.clamp(0.0, 1.0)


def _patches_to_grid(patches, grid_h, grid_w):
    N, C, H, W = patches.shape
    if N != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} patches, got {N}.")
    return patches.view(grid_h, grid_w, C, H, W).permute(2, 0, 3, 1, 4).contiguous().view(C, grid_h * H, grid_w * W)


@torch.no_grad()
def _save_normal_comparison(views, angles, gauge_angles, pred, target, valid, args, epoch):
    del angles, gauge_angles
    if not misc.is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    if views.shape[0] == 0:
        return

    sample = 0
    grid_h = args.grid_height
    grid_w = 2 * args.grid_height
    N = views.shape[1]

    pred_patches = pred.view(views.shape[0], N, 3, pred.shape[-2], pred.shape[-1])[sample]
    target_patches = target.view(views.shape[0], N, 3, target.shape[-2], target.shape[-1])[sample]
    valid_patches = valid.view(views.shape[0], N, 1, valid.shape[-2], valid.shape[-1])[sample] if valid is not None else torch.ones_like(target_patches[:, :1])

    pred_rgb_grid = _patches_to_grid(_normal_to_rgb(pred_patches, valid_patches), grid_h, grid_w)
    target_rgb_grid = _patches_to_grid(_normal_to_rgb(target_patches, valid_patches), grid_h, grid_w)

    sep_h = max(4, pred_rgb_grid.shape[-2] // 150)
    sep = torch.ones(3, sep_h, pred_rgb_grid.shape[-1], device=views.device, dtype=torch.float32)
    final = torch.cat([pred_rgb_grid, sep, target_rgb_grid], dim=1)

    compare_path = os.path.join(args.output_dir, f"normal_pred_gt_grid_epoch_{epoch:03d}.png")
    torchvision.utils.save_image(final, compare_path)
    torchvision.utils.save_image(pred_rgb_grid, os.path.join(args.output_dir, f"normal_pred_grid_epoch_{epoch:03d}.png"))
    torchvision.utils.save_image(target_rgb_grid, os.path.join(args.output_dir, f"normal_gt_grid_epoch_{epoch:03d}.png"))
    print(f"Saved grid normal pred/gt comparison to {compare_path}")


def _normal_perm_sign_candidates(device):
    import itertools
    out = []
    for perm in itertools.permutations([0, 1, 2]):
        for sign in itertools.product([-1.0, 1.0], repeat=3):
            out.append((perm, torch.tensor(sign, device=device, dtype=torch.float32)))
    return out


@torch.no_grad()
def _update_convention_sweep(sweep, pred, target, valid, args, batch_idx):
    if not getattr(args, "normal_convention_sweep", True):
        return sweep
    if batch_idx >= int(getattr(args, "normal_convention_sweep_batches", 3)):
        return sweep
    pred = F.normalize(pred.detach().float(), p=2, dim=1, eps=1e-6)
    target = F.normalize(target.detach().float(), p=2, dim=1, eps=1e-6)
    if valid is not None:
        mask = valid.detach().view(-1) > 0.5
    else:
        mask = target.norm(dim=1, keepdim=True).view(-1) > 0.1
    if mask.sum() < 10:
        return sweep
    pred_vec = pred.permute(0, 2, 3, 1).reshape(-1, 3)[mask]
    target_vec = target.permute(0, 2, 3, 1).reshape(-1, 3)[mask]
    max_pixels = int(getattr(args, "normal_convention_sweep_max_pixels", 200000))
    if pred_vec.shape[0] > max_pixels:
        idx = torch.randperm(pred_vec.shape[0], device=pred_vec.device)[:max_pixels]
        pred_vec = pred_vec[idx]
        target_vec = target_vec[idx]
    if sweep is None:
        sweep = []
        for perm, sign in _normal_perm_sign_candidates(pred_vec.device):
            sweep.append({"perm": perm, "sign": tuple(float(x) for x in sign.tolist()), "angle_sum": 0.0, "p30_count": 0.0, "count": 0})
    count = int(pred_vec.shape[0])
    for rec in sweep:
        sign = torch.tensor(rec["sign"], device=pred_vec.device, dtype=pred_vec.dtype)
        tv = target_vec[:, list(rec["perm"])] * sign.view(1, 3)
        tv = F.normalize(tv, dim=1, eps=1e-6)
        dot = (pred_vec * tv).sum(dim=1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(dot))
        rec["angle_sum"] += float(angle.sum().item())
        rec["p30_count"] += float((angle < 30.0).float().sum().item())
        rec["count"] += count
    return sweep


def _print_convention_sweep(sweep, current_p30):
    if not sweep:
        return
    best = max(sweep, key=lambda r: r["p30_count"] / max(r["count"], 1))
    p30 = 100.0 * best["p30_count"] / max(best["count"], 1)
    mean = best["angle_sum"] / max(best["count"], 1)
    print(f"[NormalConventionSweep] current P30={current_p30:.2f}% | best P30={p30:.2f}% mean={mean:.2f} deg | perm={best['perm']} sign={best['sign']}")
    if p30 > current_p30 + 10.0:
        print("[NormalConventionSweep] Large gain from a simple perm/sign transform: fix --normal_axis_perm/--normal_axis_sign/local basis before changing encoder.")


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args, epoch=0):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()
    accum = {"mean": 0.0, "median": 0.0, "rmse": 0.0, "p5": 0.0, "p7": 0.0, "p11": 0.0, "p22": 0.0, "p30": 0.0}
    total = 0
    sweep = None
    should_save_vis = (epoch + 1 == args.epochs) or bool(getattr(args, "eval", False)) or (getattr(args, "save_normal_vis_every", 0) > 0 and epoch % args.save_normal_vis_every == 0)
    saved_vis = False
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views, angles, gauge_angles, targets, valid_mask = _unpack_normal_batch(batch)
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if gauge_angles is not None:
            gauge_angles = gauge_angles.to(device, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(views, angles, gauge_angles=gauge_angles)
            pred, target, valid = _resize_normal_pred_and_targets(logits, targets, args, valid_mask)
            loss = criterion(pred, target, valid)
        metric_bidir = bool(getattr(args, "normal_metric_bidirectional", False))
        stats = compute_normal_metrics(pred.detach(), target.detach(), valid.detach() if valid is not None else None, bidirectional=metric_bidir)
        sweep = _update_convention_sweep(sweep, pred, target, valid, args, batch_idx)
        if stats["count"] > 0:
            for k in accum:
                accum[k] += stats[k] * stats["count"]
            total += stats["count"]
        if should_save_vis and not saved_vis and batch_idx == 0:
            _save_normal_comparison(views, angles, gauge_angles, pred, target, valid, args, epoch)
            saved_vis = True
        metric_logger.update(loss=loss.item())
    final = {k: (v / total if total > 0 else 0.0) for k, v in accum.items()}
    print(f"Val Normal: Mean {final['mean']:.2f} | Median {final['median']:.2f} | RMSE {final['rmse']:.2f} | P11 {final['p11']:.2f}% | P22 {final['p22']:.2f}% | P30 {final['p30']:.2f}%")
    _print_convention_sweep(sweep, final["p30"])
    return {"loss": metric_logger.loss.global_avg, **final}
