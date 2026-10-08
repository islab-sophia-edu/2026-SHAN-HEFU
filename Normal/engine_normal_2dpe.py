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
        views, angles, gauge_angles, targets, valid_mask = batch  # legacy 5-item batch; gauge ignored by the non-GCTT 2DPE model
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
            # Diagnostic/robust mode for orientation ambiguity: normal and -normal are treated equivalent.
            dot_for_loss = dot.abs()
            l1_term = torch.minimum((pred - target).abs().mean(dim=1, keepdim=True),
                                    (pred + target).abs().mean(dim=1, keepdim=True))
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
    """Resize model outputs and normal targets to the inverse-ODI patch size.

    Important detail for GT: never bilinearly resize a normal map after invalid
    pixels have been zeroed.  That mixes zero vectors into valid GT normals and
    later appears as colored contour/grid artifacts in the stitched ERP GT.  The
    target branch below therefore uses validity-aware interpolation: resize
    target*valid and valid separately, divide, then normalize.
    """
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
            logits = model(views, angles)
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
        if 'aux_loss' in locals():
            metric_logger.update(aux_loss=float(aux_loss.detach().item()) if hasattr(aux_loss, 'detach') else float(aux_loss))
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _angles_to_canonical_frame(theta, phi, psi):
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    nc = torch.stack([cos_phi * cos_theta, -cos_phi * sin_theta, sin_phi])
    x_axis = torch.stack([-sin_theta, -cos_theta, torch.zeros_like(theta)])
    y_axis = torch.stack([-sin_phi * cos_theta, sin_phi * sin_theta, cos_phi])
    c = torch.cos(psi)
    s = torch.sin(psi)
    x_g = F.normalize(c * x_axis + s * y_axis, dim=0)
    y_g = F.normalize(-s * x_axis + c * y_axis, dim=0)
    return F.normalize(nc, dim=0), x_g, y_g



def _patch_blend_weights(rr_idx, cc_idx, patch_h, patch_w, device, dtype, min_weight=1e-4):
    """Center-confidence weight for inverse-ODI stitching.

    Accepts integer or floating-point patch coordinates. The weight is high in
    the patch center and low near borders; this is critical for overlap blending
    because tangent-patch borders are where inverse projection is least stable.
    """
    if rr_idx.numel() == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if patch_h <= 1 or patch_w <= 1:
        return torch.ones_like(rr_idx, device=device, dtype=dtype)
    y = rr_idx.to(device=device, dtype=dtype) / float(max(patch_h - 1, 1))
    x = cc_idx.to(device=device, dtype=dtype) / float(max(patch_w - 1, 1))
    wy = torch.sin(math.pi * y).clamp_min(float(min_weight))
    wx = torch.sin(math.pi * x).clamp_min(float(min_weight))
    return (wx * wy).clamp_min(float(min_weight))


def _bilinear_sample_chw(patch_chw, r, c):
    """Sample a CHW patch at floating point row/col locations.

    Nearest-neighbor inverse stitching produces the visible dashed/grid seams in
    ERP. Bilinear sampling keeps the inverse projection consistent with the
    grid_sample extractor used by the dataset.
    """
    C, H, W = patch_chw.shape
    if r.numel() == 0:
        return patch_chw.new_zeros((C, 0))
    r = r.to(device=patch_chw.device, dtype=torch.float32).clamp(0.0, float(H - 1))
    c = c.to(device=patch_chw.device, dtype=torch.float32).clamp(0.0, float(W - 1))

    r0 = torch.floor(r).long().clamp(0, H - 1)
    c0 = torch.floor(c).long().clamp(0, W - 1)
    r1 = (r0 + 1).clamp(0, H - 1)
    c1 = (c0 + 1).clamp(0, W - 1)

    wr = (r - r0.float()).view(1, -1)
    wc = (c - c0.float()).view(1, -1)

    v00 = patch_chw[:, r0, c0]
    v01 = patch_chw[:, r0, c1]
    v10 = patch_chw[:, r1, c0]
    v11 = patch_chw[:, r1, c1]
    return v00 * (1.0 - wr) * (1.0 - wc) + v01 * (1.0 - wr) * wc + v10 * wr * (1.0 - wc) + v11 * wr * wc


def _relax_valid_mask_for_vis(valid_mask, kernel_size=5):
    """Dilate valid mask only for inverse-ODI visualization.

    Depth-derived normal targets use strict finite-difference validity. That
    strict mask often has one-pixel invalid bands on every tangent patch border;
    if used directly during inverse-ODI, those bands become the black grid seen
    in the saved panorama. This function fills only small border gaps for
    visualization while keeping large genuinely invalid regions invalid.
    """
    if valid_mask is None:
        return None
    k = int(kernel_size)
    if k <= 1:
        return (valid_mask.float() > 0.5).float()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    vm = valid_mask.float()
    if vm.dim() == 3:
        vm = vm.unsqueeze(1)
    return (F.max_pool2d(vm, kernel_size=k, stride=1, padding=pad) > 0.5).float()


def _fill_invalid_normals_for_vis(normals, valid_mask, kernel_size=5, iterations=1):
    """Fill only small invalid gaps in GT normal patches for visualization.

    The saved GT panorama should not be produced from zero-valued invalid pixels.
    Before using a dilated visualization mask, copy nearby valid normal vectors
    into tiny invalid bands, then renormalize.  This affects only saved images;
    loss and metrics continue to use the strict `valid` mask returned by
    `_resize_normal_pred_and_targets`.
    """
    if valid_mask is None:
        return F.normalize(normals.float(), p=2, dim=1, eps=1e-6), None

    k = int(kernel_size)
    if k <= 1 or int(iterations) <= 0:
        out = F.normalize(normals.float(), p=2, dim=1, eps=1e-6)
        return out, (valid_mask.float() > 0.5).float()
    if k % 2 == 0:
        k += 1
    pad = k // 2

    out = F.normalize(normals.float(), p=2, dim=1, eps=1e-6)
    valid = valid_mask.float()
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    valid = (valid[:, :1] > 0.5).float()

    for _ in range(int(iterations)):
        num = F.avg_pool2d(out * valid, kernel_size=k, stride=1, padding=pad) * float(k * k)
        den = F.avg_pool2d(valid, kernel_size=k, stride=1, padding=pad) * float(k * k)
        fillable = (valid < 0.5) & (den > 0.0)
        if not torch.any(fillable):
            break
        filled = F.normalize(num / den.clamp_min(1e-6), p=2, dim=1, eps=1e-6)
        out = torch.where(fillable.expand_as(out), filled, out)
        valid = torch.where(fillable, torch.ones_like(valid), valid)

    out = F.normalize(out, p=2, dim=1, eps=1e-6) * valid
    return out, valid


def _orient_normals_to_view_ray(normals_3m, ray_3m):
    """Make normals camera-facing in a canonical ERP ray convention.

    `ray_3m` is the unit ray from the panorama center to the ERP pixel. For
    Stanford depth-derived normals and most normal-estimation protocols, the
    visible surface normal should face the camera, i.e. dot(n, ray) <= 0.
    Enforcing this before overlap fusion prevents antipodal cancellation seams.
    """
    if normals_3m.numel() == 0:
        return normals_3m
    flip = (normals_3m * ray_3m).sum(dim=0, keepdim=True) > 0.0
    return torch.where(flip, -normals_3m, normals_3m)


def _accumulate_normal_vectors(canvas_flat, counts_flat, dst_idx, normal_3m, weight_m, blend_mode):
    """Accumulate or select normal vectors in vector space.

    Modes:
      average   : center-weighted vector accumulation with antipodal alignment.
                  This is the default for clean visualization because it removes
                  hard patch-switch seams.
      maxweight : keep only the source patch closest to its center. Useful as a
                  diagnostic, but it may show hard Voronoi-like seams.
      overwrite : last writer wins, kept only for debugging.
    """
    if dst_idx.numel() == 0:
        return
    if blend_mode == "average":
        existing = canvas_flat[:, dst_idx]
        existing_norm = existing.norm(dim=0, keepdim=True)
        flip = (existing_norm > 1e-6) & ((existing * normal_3m).sum(dim=0, keepdim=True) < 0.0)
        normal_3m = torch.where(flip, -normal_3m, normal_3m)
        canvas_flat[:, dst_idx] += normal_3m * weight_m.view(1, -1)
        counts_flat[:, dst_idx] += weight_m.view(1, -1)
    elif blend_mode == "maxweight":
        old_w = counts_flat[0, dst_idx]
        update = weight_m > old_w
        if torch.any(update):
            canvas_flat[:, dst_idx[update]] = normal_3m[:, update]
            counts_flat[:, dst_idx[update]] = weight_m[update].view(1, -1)
    else:  # overwrite
        canvas_flat[:, dst_idx] = normal_3m
        counts_flat[:, dst_idx] = 1.0

@torch.no_grad()
def _embed_local_normals_to_pano_inverse_odi_gpu(local_normals, angles_deg, gauge_angles_deg, pano_h, pano_w, h_fov_deg, v_fov_deg, source_valid_mask=None, blend_mode="average", local_y_sign=1.0, return_counts=False):
    if local_normals.dim() != 4 or local_normals.shape[1] != 3:
        raise ValueError(f"Expected local_normals [N,3,H,W], got {local_normals.shape}")
    if blend_mode not in {"overwrite", "average", "maxweight"}:
        raise ValueError(f"Unsupported blend_mode={blend_mode}")
    device = local_normals.device
    dtype = torch.float32
    patches = F.normalize(local_normals.float(), p=2, dim=1, eps=1e-6)
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    if gauge_angles_deg is None:
        gauge_angles_deg = torch.zeros(angles_deg.shape[0], 1, device=device, dtype=dtype)
    else:
        gauge_angles_deg = gauge_angles_deg.to(device=device, dtype=dtype)
        if gauge_angles_deg.dim() == 1:
            gauge_angles_deg = gauge_angles_deg.unsqueeze(-1)
    if source_valid_mask is not None:
        source_valid_mask = source_valid_mask.to(device=device, dtype=dtype)
        if source_valid_mask.dim() == 3:
            source_valid_mask = source_valid_mask.unsqueeze(1)
        source_valid_mask = source_valid_mask[:, :1]

    N, C, patch_h, patch_w = patches.shape
    canvas = torch.zeros(3, pano_h, pano_w, device=device, dtype=dtype)
    counts = torch.zeros(1, pano_h, pano_w, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(torch.arange(pano_h, device=device, dtype=dtype), torch.arange(pano_w, device=device, dtype=dtype), indexing="ij")
    theta_erp = (2.0 * cc / max(pano_w - 1, 1) - 1.0) * math.pi
    phi_erp = (0.5 - rr / max(pano_h - 1, 1)) * math.pi
    pn = torch.stack([torch.cos(phi_erp) * torch.cos(theta_erp), -torch.cos(phi_erp) * torch.sin(theta_erp), torch.sin(phi_erp)], dim=-1)
    h_fov = math.radians(float(h_fov_deg))
    L = (patch_w / 2.0) / math.tan(h_fov / 2.0)
    in_frustum_threshold = 2.0 * L / math.sqrt(patch_w ** 2 + patch_h ** 2 + 4.0 * L ** 2)

    for i in range(N):
        theta = torch.deg2rad(angles_deg[i, 0])
        phi = torch.deg2rad(angles_deg[i, 1])
        psi = torch.deg2rad(gauge_angles_deg[i, 0])
        nc, x_g, y_g = _angles_to_canonical_frame(theta, phi, psi)
        y_basis = float(local_y_sign) * y_g
        denom = torch.einsum("hwc,c->hw", pn, nc)
        mask = (denom > 1e-6) & (denom >= in_frustum_threshold)
        if not torch.any(mask):
            continue
        r = torch.zeros_like(denom)
        r[mask] = L / denom[mask]
        xp = r * torch.einsum("hwc,c->hw", pn, x_g)
        yp = r * torch.einsum("hwc,c->hw", pn, y_g)
        mask = mask & (xp >= -patch_w / 2.0) & (xp <= patch_w / 2.0) & (yp >= -patch_h / 2.0) & (yp <= patch_h / 2.0)
        if not torch.any(mask):
            continue
        c1 = (patch_w / 2.0 + xp - 0.5).clamp(0.0, patch_w - 1.0)
        r1 = (patch_h / 2.0 - yp - 0.5).clamp(0.0, patch_h - 1.0)
        mask_flat = mask.reshape(-1)
        r1_all = r1.reshape(-1)[mask_flat]
        c1_all = c1.reshape(-1)[mask_flat]
        rr_near_all = torch.round(r1_all).long().clamp(0, patch_h - 1)
        cc_near_all = torch.round(c1_all).long().clamp(0, patch_w - 1)
        dst_idx_all = mask_flat.nonzero(as_tuple=False).squeeze(1)
        weight_all = _patch_blend_weights(r1_all, c1_all, patch_h, patch_w, device, dtype)
        if source_valid_mask is not None:
            src_valid = source_valid_mask[i, 0, rr_near_all, cc_near_all] > 0.5
            if not torch.any(src_valid):
                continue
            r1_sel = r1_all[src_valid]
            c1_sel = c1_all[src_valid]
            dst_idx = dst_idx_all[src_valid]
            sample_w = weight_all[src_valid]
        else:
            r1_sel, c1_sel, dst_idx = r1_all, c1_all, dst_idx_all
            sample_w = weight_all
        sampled_local = _bilinear_sample_chw(patches[i], r1_sel, c1_sel)  # [3,M]
        sampled_local = F.normalize(sampled_local, dim=0, eps=1e-6)
        sampled_global = sampled_local[0:1] * x_g.view(3, 1) + sampled_local[1:2] * y_basis.view(3, 1) + sampled_local[2:3] * nc.view(3, 1)
        sampled_global = F.normalize(sampled_global, dim=0, eps=1e-6)
        ray_global = pn.reshape(-1, 3)[dst_idx].T.contiguous()
        sampled_global = _orient_normals_to_view_ray(sampled_global, ray_global)
        _accumulate_normal_vectors(canvas.view(3, -1), counts.view(1, -1), dst_idx, sampled_global, sample_w, blend_mode)
    if blend_mode == "average":
        canvas = canvas / counts.clamp_min(1e-6)
    canvas = F.normalize(canvas.unsqueeze(0), p=2, dim=1, eps=1e-6).squeeze(0)
    canvas = canvas * (counts > 0).float()
    if return_counts:
        return canvas, counts
    return canvas


@torch.no_grad()
def _embed_global_normals_to_pano_inverse_odi_gpu(global_normals, angles_deg, gauge_angles_deg, pano_h, pano_w, h_fov_deg, v_fov_deg, source_valid_mask=None, blend_mode="average", return_counts=False):
    """Inverse-ODI stitch for normal patches that are already in global/PanoMAE coordinates."""
    if global_normals.dim() != 4 or global_normals.shape[1] != 3:
        raise ValueError(f"Expected global_normals [N,3,H,W], got {global_normals.shape}")
    if blend_mode not in {"overwrite", "average", "maxweight"}:
        raise ValueError(f"Unsupported blend_mode={blend_mode}")
    device = global_normals.device
    dtype = torch.float32
    patches = F.normalize(global_normals.float(), p=2, dim=1, eps=1e-6)
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    if gauge_angles_deg is None:
        gauge_angles_deg = torch.zeros(angles_deg.shape[0], 1, device=device, dtype=dtype)
    else:
        gauge_angles_deg = gauge_angles_deg.to(device=device, dtype=dtype)
        if gauge_angles_deg.dim() == 1:
            gauge_angles_deg = gauge_angles_deg.unsqueeze(-1)
    if source_valid_mask is not None:
        source_valid_mask = source_valid_mask.to(device=device, dtype=dtype)
        if source_valid_mask.dim() == 3:
            source_valid_mask = source_valid_mask.unsqueeze(1)
        source_valid_mask = source_valid_mask[:, :1]

    N, C, patch_h, patch_w = patches.shape
    canvas = torch.zeros(3, pano_h, pano_w, device=device, dtype=dtype)
    counts = torch.zeros(1, pano_h, pano_w, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(torch.arange(pano_h, device=device, dtype=dtype), torch.arange(pano_w, device=device, dtype=dtype), indexing="ij")
    theta_erp = (2.0 * cc / max(pano_w - 1, 1) - 1.0) * math.pi
    phi_erp = (0.5 - rr / max(pano_h - 1, 1)) * math.pi
    pn = torch.stack([torch.cos(phi_erp) * torch.cos(theta_erp), -torch.cos(phi_erp) * torch.sin(theta_erp), torch.sin(phi_erp)], dim=-1)
    h_fov = math.radians(float(h_fov_deg))
    L = (patch_w / 2.0) / math.tan(h_fov / 2.0)
    in_frustum_threshold = 2.0 * L / math.sqrt(patch_w ** 2 + patch_h ** 2 + 4.0 * L ** 2)

    for i in range(N):
        theta = torch.deg2rad(angles_deg[i, 0])
        phi = torch.deg2rad(angles_deg[i, 1])
        psi = torch.deg2rad(gauge_angles_deg[i, 0])
        nc, x_g, y_g = _angles_to_canonical_frame(theta, phi, psi)
        denom = torch.einsum("hwc,c->hw", pn, nc)
        mask = (denom > 1e-6) & (denom >= in_frustum_threshold)
        if not torch.any(mask):
            continue
        r = torch.zeros_like(denom)
        r[mask] = L / denom[mask]
        xp = r * torch.einsum("hwc,c->hw", pn, x_g)
        yp = r * torch.einsum("hwc,c->hw", pn, y_g)
        mask = mask & (xp >= -patch_w / 2.0) & (xp <= patch_w / 2.0) & (yp >= -patch_h / 2.0) & (yp <= patch_h / 2.0)
        if not torch.any(mask):
            continue
        c1 = (patch_w / 2.0 + xp - 0.5).clamp(0.0, patch_w - 1.0)
        r1 = (patch_h / 2.0 - yp - 0.5).clamp(0.0, patch_h - 1.0)
        mask_flat = mask.reshape(-1)
        r1_all = r1.reshape(-1)[mask_flat]
        c1_all = c1.reshape(-1)[mask_flat]
        rr_near_all = torch.round(r1_all).long().clamp(0, patch_h - 1)
        cc_near_all = torch.round(c1_all).long().clamp(0, patch_w - 1)
        dst_idx_all = mask_flat.nonzero(as_tuple=False).squeeze(1)
        weight_all = _patch_blend_weights(r1_all, c1_all, patch_h, patch_w, device, dtype)
        if source_valid_mask is not None:
            src_valid = source_valid_mask[i, 0, rr_near_all, cc_near_all] > 0.5
            if not torch.any(src_valid):
                continue
            r1_sel = r1_all[src_valid]
            c1_sel = c1_all[src_valid]
            dst_idx = dst_idx_all[src_valid]
            sample_w = weight_all[src_valid]
        else:
            r1_sel, c1_sel, dst_idx = r1_all, c1_all, dst_idx_all
            sample_w = weight_all
        sampled = _bilinear_sample_chw(patches[i], r1_sel, c1_sel)
        sampled = F.normalize(sampled, dim=0, eps=1e-6)
        ray_global = pn.reshape(-1, 3)[dst_idx].T.contiguous()
        sampled = _orient_normals_to_view_ray(sampled, ray_global)
        _accumulate_normal_vectors(canvas.view(3, -1), counts.view(1, -1), dst_idx, sampled, sample_w, blend_mode)
    if blend_mode == "average":
        canvas = canvas / counts.clamp_min(1e-6)
    canvas = F.normalize(canvas.unsqueeze(0), p=2, dim=1, eps=1e-6).squeeze(0)
    canvas = canvas * (counts > 0).float()
    if return_counts:
        return canvas, counts
    return canvas


@torch.no_grad()
def _embed_rgb_to_pano_inverse_odi_gpu(patches, angles_deg, gauge_angles_deg, pano_h, pano_w, h_fov_deg, v_fov_deg, blend_mode="average", source_valid_mask=None, return_counts=False):
    if patches.dim() != 4:
        raise ValueError(f"Expected patches [N,C,H,W], got {patches.shape}")
    if blend_mode not in {"overwrite", "average", "maxweight"}:
        raise ValueError(f"Unsupported blend_mode={blend_mode}")
    device = patches.device
    dtype = torch.float32
    patches = patches.float().clamp(0.0, 1.0)
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    if gauge_angles_deg is None:
        gauge_angles_deg = torch.zeros(angles_deg.shape[0], 1, device=device, dtype=dtype)
    else:
        gauge_angles_deg = gauge_angles_deg.to(device=device, dtype=dtype)
        if gauge_angles_deg.dim() == 1:
            gauge_angles_deg = gauge_angles_deg.unsqueeze(-1)
    if source_valid_mask is not None:
        source_valid_mask = source_valid_mask.to(device=device, dtype=dtype)
        if source_valid_mask.dim() == 3:
            source_valid_mask = source_valid_mask.unsqueeze(1)
        source_valid_mask = source_valid_mask[:, :1]

    N, C, patch_h, patch_w = patches.shape
    canvas = torch.zeros(C, pano_h, pano_w, device=device, dtype=dtype)
    counts = torch.zeros(1, pano_h, pano_w, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(torch.arange(pano_h, device=device, dtype=dtype), torch.arange(pano_w, device=device, dtype=dtype), indexing="ij")
    theta_erp = (2.0 * cc / max(pano_w - 1, 1) - 1.0) * math.pi
    phi_erp = (0.5 - rr / max(pano_h - 1, 1)) * math.pi
    pn = torch.stack([torch.cos(phi_erp) * torch.cos(theta_erp), -torch.cos(phi_erp) * torch.sin(theta_erp), torch.sin(phi_erp)], dim=-1)
    h_fov = math.radians(float(h_fov_deg))
    L = (patch_w / 2.0) / math.tan(h_fov / 2.0)
    in_frustum_threshold = 2.0 * L / math.sqrt(patch_w ** 2 + patch_h ** 2 + 4.0 * L ** 2)

    for i in range(N):
        theta = torch.deg2rad(angles_deg[i, 0])
        phi = torch.deg2rad(angles_deg[i, 1])
        psi = torch.deg2rad(gauge_angles_deg[i, 0])
        nc, x_g, y_g = _angles_to_canonical_frame(theta, phi, psi)
        denom = torch.einsum("hwc,c->hw", pn, nc)
        mask = (denom > 1e-6) & (denom >= in_frustum_threshold)
        if not torch.any(mask):
            continue
        r = torch.zeros_like(denom)
        r[mask] = L / denom[mask]
        xp = r * torch.einsum("hwc,c->hw", pn, x_g)
        yp = r * torch.einsum("hwc,c->hw", pn, y_g)
        mask = mask & (xp >= -patch_w / 2.0) & (xp <= patch_w / 2.0) & (yp >= -patch_h / 2.0) & (yp <= patch_h / 2.0)
        if not torch.any(mask):
            continue
        c1 = (patch_w / 2.0 + xp - 0.5).clamp(0.0, patch_w - 1.0)
        r1 = (patch_h / 2.0 - yp - 0.5).clamp(0.0, patch_h - 1.0)
        mask_flat = mask.reshape(-1)
        r1_all = r1.reshape(-1)[mask_flat]
        c1_all = c1.reshape(-1)[mask_flat]
        rr_near_all = torch.round(r1_all).long().clamp(0, patch_h - 1)
        cc_near_all = torch.round(c1_all).long().clamp(0, patch_w - 1)
        dst_idx_all = mask_flat.nonzero(as_tuple=False).squeeze(1)
        weight_all = _patch_blend_weights(r1_all, c1_all, patch_h, patch_w, device, dtype)
        if source_valid_mask is not None:
            src_valid = source_valid_mask[i, 0, rr_near_all, cc_near_all] > 0.5
            if not torch.any(src_valid):
                continue
            r1_sel = r1_all[src_valid]
            c1_sel = c1_all[src_valid]
            dst_idx = dst_idx_all[src_valid]
            sample_w = weight_all[src_valid]
        else:
            r1_sel, c1_sel, dst_idx = r1_all, c1_all, dst_idx_all
            sample_w = weight_all
        sampled = _bilinear_sample_chw(patches[i], r1_sel, c1_sel)
        canvas_flat = canvas.view(C, -1)
        counts_flat = counts.view(1, -1)
        if blend_mode == "average":
            canvas_flat[:, dst_idx] += sampled * sample_w.view(1, -1)
            counts_flat[:, dst_idx] += sample_w.view(1, -1)
        elif blend_mode == "maxweight":
            old_w = counts_flat[0, dst_idx]
            update = sample_w > old_w
            if torch.any(update):
                canvas_flat[:, dst_idx[update]] = sampled[:, update]
                counts_flat[:, dst_idx[update]] = sample_w[update].view(1, -1)
        else:
            canvas_flat[:, dst_idx] = sampled
            counts_flat[:, dst_idx] = 1.0
    if blend_mode == "average":
        canvas = canvas / counts.clamp_min(1e-6)
    canvas = canvas.clamp(0.0, 1.0)
    if return_counts:
        return canvas, counts
    return canvas


def _normal_to_rgb(normal, valid_mask=None):
    rgb = (normal.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    if valid_mask is not None:
        rgb = rgb * (valid_mask.float() > 0.5).float()
    return rgb.clamp(0.0, 1.0)


def _jet_colormap(x, valid_mask=None):
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif x.dim() == 3:
        x = x[:1].unsqueeze(0)
        squeeze = True
    else:
        x = x[:, :1]
        squeeze = False
    x = x.float().clamp(0.0, 1.0)
    r = torch.zeros_like(x); g = torch.zeros_like(x); b = torch.zeros_like(x)
    m = (x >= 0.0) & (x < 0.25); b = torch.where(m, torch.ones_like(b), b); g = torch.where(m, x / 0.25, g)
    m = (x >= 0.25) & (x < 0.50); b = torch.where(m, 1.0 - (x - 0.25) / 0.25, b); g = torch.where(m, torch.ones_like(g), g)
    m = (x >= 0.50) & (x < 0.75); r = torch.where(m, (x - 0.50) / 0.25, r); g = torch.where(m, torch.ones_like(g), g)
    m = x >= 0.75; r = torch.where(m, torch.ones_like(r), r); g = torch.where(m, 1.0 - (x - 0.75) / 0.25, g)
    rgb = torch.cat([r, g.clamp(0.0, 1.0), b], dim=1).clamp(0.0, 1.0)
    if valid_mask is not None:
        vm = valid_mask if valid_mask.dim() == 4 else valid_mask.unsqueeze(0)
        rgb = rgb * (vm[:, :1].float() > 0.5).float()
    return rgb.squeeze(0) if squeeze else rgb


def _patches_to_grid(patches, grid_h, grid_w):
    N, C, H, W = patches.shape
    return patches.view(grid_h, grid_w, C, H, W).permute(2, 0, 3, 1, 4).contiguous().view(C, grid_h * H, grid_w * W)


@torch.no_grad()
def _save_normal_comparison(views, angles, gauge_angles, pred, target, valid, args, epoch):
    if not misc.is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    sample = 0
    grid_h = args.grid_height
    grid_w = 2 * args.grid_height
    N = views.shape[1]

    pred_patches = pred.view(views.shape[0], N, 3, pred.shape[-2], pred.shape[-1])[sample]
    target_patches = target.view(views.shape[0], N, 3, target.shape[-2], target.shape[-1])[sample]
    valid_patches = valid.view(views.shape[0], N, 1, valid.shape[-2], valid.shape[-1])[sample] if valid is not None else torch.ones_like(target_patches[:, :1])

    angles_sample = angles[sample]
    gauge_sample = gauge_angles[sample] if gauge_angles is not None else None

    # Visualization-only seam fix.  A tiny FOV expansion plus bilinear inverse
    # sampling removes exact cell-boundary holes.  GT uses an independent support
    # mask and single-source center selection to avoid averaging inconsistent
    # depth-derived normal targets from overlapping tangent views.
    h_fov = 360.0 / grid_w
    v_fov = 180.0 / grid_h
    fov_scale = float(getattr(args, "inverse_odi_fov_scale", 1.02))
    fov_scale = max(1.0, min(fov_scale, 1.15))
    h_fov_vis = h_fov * fov_scale
    v_fov_vis = v_fov * fov_scale

    pred_blend_mode = getattr(args, "inverse_odi_blend", "average")
    # For GT, maxweight is safer than average: neighboring tangent patches can
    # have slightly different depth-derived normals after perspective sampling,
    # and averaging them creates the visible contour/ripple artifacts.
    gt_blend_mode = getattr(args, "inverse_odi_gt_blend", "maxweight")
    local_y_sign = float(getattr(args, "normal_local_y_sign", 1.0))

    pred_valid_dilate = int(getattr(args, "inverse_odi_valid_dilate", 5))
    gt_valid_dilate = int(getattr(args, "inverse_odi_gt_valid_dilate", max(pred_valid_dilate, 5)))
    gt_fill_kernel = int(getattr(args, "inverse_odi_gt_fill_kernel", 5))
    gt_fill_iters = int(getattr(args, "inverse_odi_gt_fill_iters", 1))

    stitch_valid_pred = _relax_valid_mask_for_vis(valid_patches, pred_valid_dilate)
    target_patches_vis, target_valid_filled = _fill_invalid_normals_for_vis(
        target_patches, valid_patches, kernel_size=gt_fill_kernel, iterations=gt_fill_iters)
    stitch_valid_target = _relax_valid_mask_for_vis(
        target_valid_filled if target_valid_filled is not None else valid_patches, gt_valid_dilate)

    target_frame = str(getattr(args, "normal_target_frame", "global")).lower()
    if target_frame == "global":
        pred_world, pred_counts = _embed_global_normals_to_pano_inverse_odi_gpu(
            pred_patches, angles_sample, gauge_sample, args.pano_h, args.pano_w, h_fov_vis, v_fov_vis,
            source_valid_mask=stitch_valid_pred, blend_mode=pred_blend_mode, return_counts=True)
        target_world, target_counts = _embed_global_normals_to_pano_inverse_odi_gpu(
            target_patches_vis, angles_sample, gauge_sample, args.pano_h, args.pano_w, h_fov_vis, v_fov_vis,
            source_valid_mask=stitch_valid_target, blend_mode=gt_blend_mode, return_counts=True)
    else:
        pred_world, pred_counts = _embed_local_normals_to_pano_inverse_odi_gpu(
            pred_patches, angles_sample, gauge_sample, args.pano_h, args.pano_w, h_fov_vis, v_fov_vis,
            source_valid_mask=stitch_valid_pred, blend_mode=pred_blend_mode, local_y_sign=local_y_sign, return_counts=True)
        target_world, target_counts = _embed_local_normals_to_pano_inverse_odi_gpu(
            target_patches_vis, angles_sample, gauge_sample, args.pano_h, args.pano_w, h_fov_vis, v_fov_vis,
            source_valid_mask=stitch_valid_target, blend_mode=gt_blend_mode, local_y_sign=local_y_sign, return_counts=True)

    # Render pred and GT with independent inverse-ODI support.  Using pred∩GT
    # validity makes GT inherit pred holes and creates false black seams.
    pred_rgb = _normal_to_rgb(pred_world, pred_counts > 0)
    target_rgb = _normal_to_rgb(target_world, target_counts > 0)

    sep_h = max(4, args.pano_h // 150)
    sep = torch.ones(3, sep_h, pred_rgb.shape[-1], device=views.device)
    final = torch.cat([pred_rgb, sep, target_rgb], dim=1)

    compare_path = os.path.join(args.output_dir, f"normal_pred_gt_inverse_odi_epoch_{epoch:03d}.png")
    torchvision.utils.save_image(final, compare_path)
    torchvision.utils.save_image(pred_rgb, os.path.join(args.output_dir, f"normal_pred_inverse_odi_epoch_{epoch:03d}.png"))
    torchvision.utils.save_image(target_rgb, os.path.join(args.output_dir, f"normal_gt_inverse_odi_epoch_{epoch:03d}.png"))
    print(f"Saved inverse-ODI normal pred/gt comparison to {compare_path}")

def _normal_perm_sign_candidates(device):
    import itertools
    out = []
    for perm in itertools.permutations([0, 1, 2]):
        for sign in itertools.product([-1.0, 1.0], repeat=3):
            out.append((perm, torch.tensor(sign, device=device, dtype=torch.float32)))
    return out


@torch.no_grad()
def _update_convention_sweep(sweep, pred, target, valid, args, batch_idx):
    """Validation-only diagnostic: if a simple channel permutation/sign change gives
    much higher P30, the bottleneck is coordinate convention, not encoder capacity.
    """
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
        views = views.to(device, non_blocking=True); angles = angles.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
        if gauge_angles is not None: gauge_angles = gauge_angles.to(device, non_blocking=True)
        if valid_mask is not None: valid_mask = valid_mask.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(views, angles)
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
