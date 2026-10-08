import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import util.misc as misc


def _unpack_depth_batch(batch):
    """
    2DPE depth batch unpacking.

    Preferred 2DPE dataset:
        views, angles, targets, valid_mask

    Fallback old format:
        views, angles, targets

    If a 5-item batch is accidentally passed, the third tensor is ignored so
    this engine remains non-GCTT/2DPE-compatible and never consumes gauge_angles.
    """
    if len(batch) == 5:
        views, angles, _ignored_third_tensor, targets, valid_mask = batch
    elif len(batch) == 4:
        views, angles, targets, valid_mask = batch
    elif len(batch) == 3:
        views, angles, targets = batch
        valid_mask = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}. Expected 3, 4 or 5.")
    return views, angles, targets, valid_mask

class DepthLoss(nn.Module):
    """
    Minimal masked L1 depth loss, aligned with the normal task's simple loss style.

    Normal uses one direct masked criterion over valid pixels.  For depth, the
    closest equivalent is masked L1 on normalized depth:

        loss = mean(abs(pred_norm - target_norm)) over valid pixels

    Why L1 rather than SILog/L2:
      - SILog is useful, but it is log-scale / relative-depth supervision and
        adds an extra scale-invariant assumption. That is less direct than the
        normal task's simple per-pixel criterion.
      - L2 over-penalizes large metric outliers, especially with max_depth=100m,
        and can make training dominated by a few far/noisy pixels.
      - L1 is simple, robust, and keeps the same normalized-depth target used by
        the dataset: target_norm = depth_meters / max_depth.

    This class accepts the old mixed-loss keyword arguments only for backward
    compatibility with existing main_finetune_depth_*.py files. They are ignored.
    There is no SILog, no BerHu, no gradient loss, no RMSE term, no far-depth
    reweighting, and no bin classification loss.
    """

    def __init__(
        self,
        max_depth=100.0,
        min_depth=0.05,
        w_l1=1.0,
        valid_eps=1e-4,
        **unused_kwargs,
    ):
        super().__init__()
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)
        self.w_l1 = float(w_l1)
        self.valid_eps = float(valid_eps)

    def forward(self, pred, target, valid_mask=None, depth_bin_logits=None, depth_bin_edges_m=None):
        del depth_bin_logits, depth_bin_edges_m

        pred_f = pred.float().clamp(0.0, 1.0)
        target_f = target.float().clamp(0.0, 1.0)

        target_m = target_f * self.max_depth
        mask = target_m > max(self.min_depth, self.valid_eps)
        if valid_mask is not None:
            mask = mask & (valid_mask.float() > 0.5)

        if mask.sum() < 10:
            return (pred_f * 0.0).sum()

        return self.w_l1 * torch.abs(pred_f - target_f)[mask].mean()

def compute_depth_metrics(pred_all, target_all, max_depth: float):
    """
    pred_all, target_all: 1D tensors in meters.
    """
    pred_all = pred_all.float()
    target_all = target_all.float()

    mask = (target_all > 1e-3) & (target_all <= max_depth)
    pred = pred_all[mask].clamp(min=1e-4, max=max_depth)
    target = target_all[mask].clamp(min=1e-4, max=max_depth)

    if pred.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    thresh = torch.max(target / pred, pred / target)
    delta1 = (thresh < 1.25).float().mean()

    diff = pred - target
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff ** 2).mean())
    abs_rel = (diff.abs() / target).mean()

    return mae.item(), rmse.item(), abs_rel.item(), delta1.item()


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


def _resize_depth_pred_and_targets(logits, targets, args, valid_mask=None):
    B, N, C, H, W = logits.shape
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)

    pred_resized = F.interpolate(
        logits.view(B * N, C, H, W),
        size=(patch_h, patch_w),
        mode="bilinear",
        align_corners=False,
    )

    targets_resized = F.interpolate(
        targets.view(B * N, 1, targets.shape[-2], targets.shape[-1]).float(),
        size=(patch_h, patch_w),
        mode="bilinear",
        align_corners=False,
    )

    valid_resized = None
    if valid_mask is not None:
        valid_resized = F.interpolate(
            valid_mask.view(B * N, 1, valid_mask.shape[-2], valid_mask.shape[-1]).float(),
            size=(patch_h, patch_w),
            mode="nearest",
        )
        targets_resized = targets_resized * (valid_resized > 0.5).float()

    return pred_resized.clamp(0.0, 1.0), targets_resized.clamp(0.0, 1.0), valid_resized



def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_last_depth_bin_logits(model):
    return getattr(_unwrap_model(model), "last_depth_bin_logits", None)


def _get_depth_bin_edges_m(model):
    m = _unwrap_model(model)
    head = getattr(m, "depth_head", None)
    return getattr(head, "bin_edges_m", None)


def _resize_depth_bin_logits(bin_logits, args):
    if bin_logits is None:
        return None
    B, N, K, H, W = bin_logits.shape
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)
    logits = bin_logits.view(B * N, K, H, W)
    if (H, W) != (patch_h, patch_w):
        logits = F.interpolate(logits, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
    return logits


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", misc.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()
    num_steps_per_epoch = len(data_loader)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        views, angles, targets, valid_mask = _unpack_depth_batch(batch)

        global_step = epoch * num_steps_per_epoch + data_iter_step
        max_steps = args.epochs * num_steps_per_epoch
        adjust_learning_rate(optimizer, epoch, args, global_step, max_steps)

        is_last_batch = data_iter_step == len(data_loader) - 1
        is_update_step = ((data_iter_step + 1) % args.accum_iter == 0) or is_last_batch

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(views, angles)
            pred_resized, targets_resized, valid_resized = _resize_depth_pred_and_targets(logits, targets, args, valid_mask=valid_mask)
            depth_bin_logits = _resize_depth_bin_logits(_get_last_depth_bin_logits(model), args)
            depth_bin_edges_m = _get_depth_bin_edges_m(model)
            loss = criterion(
                pred_resized,
                targets_resized,
                valid_mask=valid_resized,
                depth_bin_logits=depth_bin_logits,
                depth_bin_edges_m=depth_bin_edges_m,
            )

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


def _patches_to_grid(patches, grid_h, grid_w):
    """
    Fallback visualization only: patches [N, C, H, W] -> [C, grid_h*H, grid_w*W].
    The main validation visualization below uses inverse ODI projection instead.
    """
    N, C, H, W = patches.shape
    assert N == grid_h * grid_w, f"Expected {grid_h * grid_w} patches, got {N}."
    x = patches.view(grid_h, grid_w, C, H, W)
    x = x.permute(2, 0, 3, 1, 4).contiguous()
    return x.view(C, grid_h * H, grid_w * W)


def _jet_colormap(depth_norm, valid_mask=None):
    """
    Apply a JET-like pseudocolor map in RGB order.

    Contract:
        near / small depth  -> blue
        middle depth        -> cyan / green / yellow
        far / large depth   -> red
        invalid             -> black

    depth_norm: [N,1,H,W], [1,H,W], or [H,W], normalized to [0,1].
    valid_mask: optional same spatial shape, where >0.5 means valid.

    Important: this function must be applied AFTER inverse-ODI projection when
    making panorama visualizations. If color is applied patch-by-patch before
    inverse projection, invalid black pixels from an overlapping patch can
    overwrite valid colored pixels from another patch.
    """
    squeeze_mode = None
    if depth_norm.dim() == 2:      # [H,W]
        x = depth_norm.unsqueeze(0).unsqueeze(0)
        squeeze_mode = "hw"
    elif depth_norm.dim() == 3:    # [1,H,W] or [C,H,W]
        x = depth_norm[:1].unsqueeze(0)
        squeeze_mode = "chw"
    elif depth_norm.dim() == 4:    # [B,1,H,W] or [N,1,H,W]
        x = depth_norm[:, :1]
        squeeze_mode = None
    else:
        raise ValueError(f"Unexpected depth tensor shape: {depth_norm.shape}")

    x = x.float().clamp(0.0, 1.0)

    # Piecewise linear JET in RGB order. Endpoints are full blue/full red, not
    # grayscale. This is intentionally close to OpenCV COLORMAP_JET semantics,
    # but implemented in torch to avoid a cv2 dependency in training.
    r = torch.zeros_like(x)
    g = torch.zeros_like(x)
    b = torch.zeros_like(x)

    # [0, 0.25]: blue -> cyan
    m = (x >= 0.0) & (x < 0.25)
    b = torch.where(m, torch.ones_like(b), b)
    g = torch.where(m, x / 0.25, g)

    # [0.25, 0.50]: cyan -> green
    m = (x >= 0.25) & (x < 0.50)
    b = torch.where(m, 1.0 - (x - 0.25) / 0.25, b)
    g = torch.where(m, torch.ones_like(g), g)

    # [0.50, 0.75]: green -> yellow
    m = (x >= 0.50) & (x < 0.75)
    r = torch.where(m, (x - 0.50) / 0.25, r)
    g = torch.where(m, torch.ones_like(g), g)

    # [0.75, 1.00]: yellow -> red
    m = x >= 0.75
    r = torch.where(m, torch.ones_like(r), r)
    g = torch.where(m, 1.0 - (x - 0.75) / 0.25, g)

    rgb = torch.cat([r, g.clamp(0.0, 1.0), b], dim=1).clamp(0.0, 1.0)

    if valid_mask is not None:
        vm = valid_mask
        if vm.dim() == 2:
            vm = vm.unsqueeze(0).unsqueeze(0)
        elif vm.dim() == 3:
            vm = vm[:1].unsqueeze(0) if squeeze_mode == "chw" else vm.unsqueeze(1)
        elif vm.dim() == 4:
            vm = vm[:, :1]
        else:
            raise ValueError(f"Unexpected valid_mask shape: {valid_mask.shape}")
        vm = vm.to(device=rgb.device, dtype=rgb.dtype)
        rgb = rgb * (vm > 0.5).float()

    if squeeze_mode == "hw" or squeeze_mode == "chw":
        return rgb.squeeze(0)
    return rgb


# Backward-compatible alias used by old optional grid visualization code.
def _depth_to_rgb(depth_norm, valid_mask=None):
    return _jet_colormap(depth_norm, valid_mask)


def _scalar_stats(x: torch.Tensor, valid_mask: torch.Tensor):
    """Return min/max/p95/p99 for logging/debugging; values stay on device."""
    if valid_mask is None:
        vals = x.reshape(-1).float()
    else:
        vals = x[valid_mask > 0].reshape(-1).float()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        z = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return z, z, z, z
    return (
        vals.min(),
        vals.max(),
        torch.quantile(vals, 0.95),
        torch.quantile(vals, 0.99),
    )


def _choose_depth_vis_range(target_m_pano, target_valid_pano, args):
    """
    Choose the color range for depth visualization only.

    This deliberately does NOT change training/evaluation cliff.  For indoor
    Stanford2D3D panoramas, almost all valid depth may be far below 100 m.
    If the JET map is normalized by max_depth=100, everything appears blue,
    even when the scalar depth is correct.  The default dynamic mode maps the
    farthest valid depths in the current panorama to red so visual inspection is
    useful, while metrics still use args.max_depth.
    """
    mode = getattr(args, "vis_depth_norm", "dynamic")
    max_depth = float(getattr(args, "max_depth", 100.0))
    user_vmin = float(getattr(args, "vis_depth_min", 0.0))
    user_vmax = float(getattr(args, "vis_depth_max", 0.0))

    if mode == "fixed":
        vmin = torch.tensor(user_vmin, device=target_m_pano.device, dtype=torch.float32)
        vmax_value = user_vmax if user_vmax > user_vmin else max_depth
        vmax = torch.tensor(vmax_value, device=target_m_pano.device, dtype=torch.float32)
        return vmin, vmax

    vals = target_m_pano[target_valid_pano > 0].reshape(-1).float()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return (
            torch.tensor(0.0, device=target_m_pano.device, dtype=torch.float32),
            torch.tensor(max_depth, device=target_m_pano.device, dtype=torch.float32),
        )

    percentile = float(getattr(args, "vis_depth_percentile", 99.0))
    percentile = max(50.0, min(100.0, percentile)) / 100.0
    # Keep near = blue anchored at user_vmin/default 0.  Only vmax is dynamic.
    vmin = torch.tensor(user_vmin, device=target_m_pano.device, dtype=torch.float32)
    vmax = torch.quantile(vals.clamp(min=user_vmin), percentile)
    # Numerical safety and a floor so extremely flat depth maps do not explode.
    vmax = torch.maximum(vmax, vmin + torch.tensor(1.0, device=target_m_pano.device))
    return vmin, vmax


def _metric_depth_to_jet(depth_m_pano, valid_pano, vmin_m, vmax_m):
    depth_norm = (depth_m_pano.float() - vmin_m) / (vmax_m - vmin_m + 1e-6)
    return _jet_colormap(depth_norm.clamp(0.0, 1.0), valid_pano)


def _choose_error_vis_max(error_m_pano, error_valid_pano, args):
    user_vmax = float(getattr(args, "vis_error_max", 0.0))
    if user_vmax > 0:
        return torch.tensor(user_vmax, device=error_m_pano.device, dtype=torch.float32)
    vals = error_m_pano[error_valid_pano > 0].reshape(-1).float()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=error_m_pano.device, dtype=torch.float32)
    p = float(getattr(args, "vis_error_percentile", 99.0))
    p = max(50.0, min(100.0, p)) / 100.0
    vmax = torch.quantile(vals, p)
    return torch.maximum(vmax, torch.tensor(0.1, device=error_m_pano.device, dtype=torch.float32))

def _angles_to_canonical_frame(theta, phi, psi):
    """
    Returns nc, x_g, y_g for one patch, following your canonical tangent-frame convention.
    theta/lon, phi/lat, psi/zero-gauge canonical frame angles are scalar radians.
    """
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    nc = torch.stack([cos_phi * cos_theta, -cos_phi * sin_theta, sin_phi])
    x_axis = torch.stack([-sin_theta, -cos_theta, torch.zeros_like(theta)])
    y_axis = torch.stack([-sin_phi * cos_theta, sin_phi * sin_theta, cos_phi])

    c = torch.cos(psi)
    s = torch.sin(psi)
    x_g = c * x_axis + s * y_axis
    y_g = -s * x_axis + c * y_axis

    nc = F.normalize(nc, dim=0)
    x_g = F.normalize(x_g, dim=0)
    y_g = F.normalize(y_g, dim=0)
    return nc, x_g, y_g


@torch.no_grad()
def _embed_patches_to_pano_inverse_odi_gpu(
    patches,
    angles_deg,
    gauge_angles_deg,
    pano_h,
    pano_w,
    h_fov_deg,
    v_fov_deg,
    blend_mode="overwrite",
    source_valid_mask=None,
    return_counts=False,
    clamp_output=True,
):
    """
    Inverse of the dataset's tangent/ODI extraction, with zero gauge for the non-GCTT path.

    Args:
        patches: [N, C, H, W], scalar or RGB patches in visualization range.
        angles_deg: [N, 2], [lon/theta, lat/phi] in degrees.
        gauge_angles_deg: kept for internal compatibility; 2DPE passes None, i.e. zero gauge.
        source_valid_mask: optional [N,1,H,W]. Invalid source pixels are NOT
            written to the panorama. This is critical for depth visualization:
            invalid black pixels must not overwrite valid pixels from an
            overlapping patch.
        return_counts: if True, returns (pano, counts), where counts indicates
            projected valid coverage per ERP pixel.
        clamp_output: True for RGB/image tensors in [0,1]. False for metric
            scalar maps such as depth in meters or absolute error in meters.
            Depth visualization must not clamp metric scalar values before color
            mapping; otherwise every value above 1m becomes saturated.

    Returns:
        pano: [C, pano_h, pano_w]
    """
    if patches.dim() != 4:
        raise ValueError(f"Expected patches [N,C,H,W], got {patches.shape}.")
    if blend_mode not in {"overwrite", "average"}:
        raise ValueError(f"blend_mode must be 'overwrite' or 'average', got {blend_mode}.")

    device = patches.device
    dtype = torch.float32
    patches = patches.float()
    if clamp_output:
        patches = patches.clamp(0.0, 1.0)
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

    # ERP pixel centers -> spherical rays, consistent with grid_sample align_corners=True.
    rr, cc = torch.meshgrid(
        torch.arange(pano_h, device=device, dtype=dtype),
        torch.arange(pano_w, device=device, dtype=dtype),
        indexing="ij",
    )
    theta_erp = (2.0 * cc / max(pano_w - 1, 1) - 1.0) * math.pi
    phi_erp = (0.5 - rr / max(pano_h - 1, 1)) * math.pi
    pn = torch.stack(
        [
            torch.cos(phi_erp) * torch.cos(theta_erp),
            -torch.cos(phi_erp) * torch.sin(theta_erp),
            torch.sin(phi_erp),
        ],
        dim=-1,
    )  # [H,W,3]

    h_fov = math.radians(float(h_fov_deg))
    # The dataset extraction uses horizontal FoV to compute focal length.
    L = (patch_w / 2.0) / math.tan(h_fov / 2.0)
    in_frustum_threshold = 2.0 * L / math.sqrt(patch_w ** 2 + patch_h ** 2 + 4.0 * L ** 2)

    for i in range(N):
        theta = torch.deg2rad(angles_deg[i, 0])
        phi = torch.deg2rad(angles_deg[i, 1])
        psi = torch.deg2rad(gauge_angles_deg[i, 0])
        nc, x_g, y_g = _angles_to_canonical_frame(theta, phi, psi)

        denom = torch.einsum("hwc,c->hw", pn, nc)
        visible = denom > 1e-6
        mask = visible & (denom >= in_frustum_threshold)

        if not torch.any(mask):
            continue

        r = torch.zeros_like(denom)
        r[mask] = L / denom[mask]
        xp = r * torch.einsum("hwc,c->hw", pn, x_g)
        yp = r * torch.einsum("hwc,c->hw", pn, y_g)

        mask = mask & (xp > -patch_w / 2.0) & (xp < patch_w / 2.0) & (yp > -patch_h / 2.0) & (yp < patch_h / 2.0)
        if not torch.any(mask):
            continue

        c1 = patch_w / 2.0 + xp - 0.5
        r1 = patch_h / 2.0 - yp - 0.5
        c1_idx = torch.round(c1).long().clamp(0, patch_w - 1)
        r1_idx = torch.round(r1).long().clamp(0, patch_h - 1)

        mask_flat = mask.reshape(-1)
        rr_idx_all = r1_idx.reshape(-1)[mask_flat]
        cc_idx_all = c1_idx.reshape(-1)[mask_flat]
        dst_idx_all = mask_flat.nonzero(as_tuple=False).squeeze(1)

        # Skip invalid source depth pixels instead of writing black to the canvas.
        if source_valid_mask is not None:
            src_valid = source_valid_mask[i, 0, rr_idx_all, cc_idx_all] > 0.5
            if not torch.any(src_valid):
                continue
            rr_idx = rr_idx_all[src_valid]
            cc_idx = cc_idx_all[src_valid]
            dst_idx = dst_idx_all[src_valid]
        else:
            rr_idx = rr_idx_all
            cc_idx = cc_idx_all
            dst_idx = dst_idx_all

        sampled = patches[i, :, rr_idx, cc_idx]  # [C, M]
        if sampled.numel() == 0:
            continue

        canvas_flat = canvas.view(C, -1)
        counts_flat = counts.view(1, -1)

        if blend_mode == "average":
            canvas_flat[:, dst_idx] += sampled
            counts_flat[:, dst_idx] += 1.0
        else:
            canvas_flat[:, dst_idx] = sampled
            counts_flat[:, dst_idx] = 1.0

    if blend_mode == "average":
        canvas = canvas / counts.clamp_min(1.0)

    if clamp_output:
        canvas = canvas.clamp(0.0, 1.0)
    if return_counts:
        return canvas, counts
    return canvas

def _save_depth_comparison(views, angles, pred_resized, targets_resized, valid_resized, args, epoch):
    """
    Save pretrain-style diagnostic visualization using inverse ODI projection:
        original RGB ERP
        predicted depth ERP, pseudocolor
        ground-truth depth ERP, pseudocolor
        absolute error ERP, pseudocolor

    Critical implementation detail:
        scalar depth is inverse-projected first; JET color is applied after the
        ERP scalar map is formed. This avoids wrong colors caused by averaging or
        overwriting already-colorized patches. Invalid depth pixels are also
        skipped during projection instead of being written as black patches.
    """
    if not misc.is_main_process():
        return

    os.makedirs(args.output_dir, exist_ok=True)

    B = views.shape[0]
    if B == 0:
        return

    sample = 0
    grid_h = args.grid_height
    grid_w = 2 * args.grid_height
    max_depth = float(getattr(args, "max_depth", 100.0))
    vis_depth_max = float(getattr(args, "vis_depth_max", 0.0))
    if vis_depth_max <= 0.0:
        vis_depth_max = max_depth

    mean = torch.tensor([0.485, 0.456, 0.406], device=views.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=views.device).view(1, 3, 1, 1)

    N = views.shape[1]
    rgb_patches = torch.clamp(views[sample, :, :3] * std + mean, 0.0, 1.0)
    pred_norm = pred_resized.view(B, N, 1, pred_resized.shape[-2], pred_resized.shape[-1])[sample].float().clamp(0.0, 1.0)
    target_norm = targets_resized.view(B, N, 1, targets_resized.shape[-2], targets_resized.shape[-1])[sample].float().clamp(0.0, 1.0)
    if valid_resized is not None:
        valid_norm = valid_resized.view(B, N, 1, valid_resized.shape[-2], valid_resized.shape[-1])[sample].float()
        valid_norm = (valid_norm > 0.5).float()
    else:
        valid_norm = (target_norm > 1e-6).float()

    # Convert to metric depth first.  For correct color, inverse-project scalar
    # metric depth, then normalize the final ERP scalar map for JET.  Do NOT
    # normalize each patch by max_depth before inverse projection: with
    # cliff=100m most indoor Stanford depths are only a small fraction of the
    # range, so a fixed [0,100] color scale will make everything blue.
    pred_m = pred_norm * max_depth
    target_m = target_norm * max_depth
    error_m = torch.abs(pred_m - target_m)

    angles_sample = angles[sample]
    gauge_sample = None

    h_fov = 360.0 / grid_w
    v_fov = 180.0 / grid_h
    blend_mode = getattr(args, "inverse_odi_blend", "average")

    rgb_pano = _embed_patches_to_pano_inverse_odi_gpu(
        rgb_patches, angles_sample, gauge_sample,
        args.pano_h, args.pano_w, h_fov, v_fov, blend_mode=blend_mode,
        clamp_output=True,
    )

    # Project scalar metric depth/validity first. Do NOT project already-colored
    # depth. Invalid source pixels are skipped inside the inverse projector.
    pred_scalar_pano_m, pred_counts = _embed_patches_to_pano_inverse_odi_gpu(
        pred_m, angles_sample, gauge_sample,
        args.pano_h, args.pano_w, h_fov, v_fov,
        blend_mode=blend_mode,
        source_valid_mask=valid_norm,
        return_counts=True,
        clamp_output=False,
    )
    target_scalar_pano_m, target_counts = _embed_patches_to_pano_inverse_odi_gpu(
        target_m, angles_sample, gauge_sample,
        args.pano_h, args.pano_w, h_fov, v_fov,
        blend_mode=blend_mode,
        source_valid_mask=valid_norm,
        return_counts=True,
        clamp_output=False,
    )
    error_scalar_pano_m, error_counts = _embed_patches_to_pano_inverse_odi_gpu(
        error_m, angles_sample, gauge_sample,
        args.pano_h, args.pano_w, h_fov, v_fov,
        blend_mode=blend_mode,
        source_valid_mask=valid_norm,
        return_counts=True,
        clamp_output=False,
    )

    pred_valid_pano = pred_counts > 0
    target_valid_pano = target_counts > 0
    error_valid_pano = error_counts > 0

    # Shared depth color range: use GT panorama statistics so pred and GT are
    # visually comparable. Default is dynamic p99 normalization for visualization
    # only; metrics/training still use max_depth/cliff=100.
    depth_vmin_m, depth_vmax_m = _choose_depth_vis_range(target_scalar_pano_m, target_valid_pano, args)
    error_vmax_m = _choose_error_vis_max(error_scalar_pano_m, error_valid_pano, args)

    pred_pano = _metric_depth_to_jet(pred_scalar_pano_m, pred_valid_pano, depth_vmin_m, depth_vmax_m)
    target_pano = _metric_depth_to_jet(target_scalar_pano_m, target_valid_pano, depth_vmin_m, depth_vmax_m)
    error_pano = _metric_depth_to_jet(error_scalar_pano_m, error_valid_pano, torch.tensor(0.0, device=views.device), error_vmax_m)

    t_min, t_max, t_p95, t_p99 = _scalar_stats(target_scalar_pano_m, target_valid_pano)
    p_min, p_max, p_p95, p_p99 = _scalar_stats(pred_scalar_pano_m, pred_valid_pano)
    print(
        f"[DepthVis] color_norm={getattr(args, 'vis_depth_norm', 'dynamic')} "
        f"range=[{depth_vmin_m.item():.3f},{depth_vmax_m.item():.3f}]m | "
        f"GT min/max/p95/p99={t_min.item():.3f}/{t_max.item():.3f}/{t_p95.item():.3f}/{t_p99.item():.3f}m | "
        f"Pred min/max/p95/p99={p_min.item():.3f}/{p_max.item():.3f}/{p_p95.item():.3f}/{p_p99.item():.3f}m"
    )

    sep_h = max(4, args.pano_h // 150)
    sep = torch.ones(3, sep_h, args.pano_w, device=views.device, dtype=torch.float32)
    final = torch.cat([rgb_pano, sep, pred_pano, sep, target_pano, sep, error_pano], dim=1)

    compare_path = os.path.join(args.output_dir, f"depth_inverse_odi_compare_epoch_{epoch:03d}.png")
    torchvision.utils.save_image(final, compare_path)
    torchvision.utils.save_image(rgb_pano, os.path.join(args.output_dir, f"rgb_inverse_odi_epoch_{epoch:03d}.png"))
    torchvision.utils.save_image(pred_pano, os.path.join(args.output_dir, f"depth_pred_inverse_odi_epoch_{epoch:03d}_max{max_depth:g}m.png"))
    torchvision.utils.save_image(target_pano, os.path.join(args.output_dir, f"depth_gt_inverse_odi_epoch_{epoch:03d}_max{max_depth:g}m.png"))
    torchvision.utils.save_image(error_pano, os.path.join(args.output_dir, f"depth_error_inverse_odi_epoch_{epoch:03d}_max{max_depth:g}m.png"))

    # Also save scalar normalized maps for debugging the colormap itself.
    # Debug scalar maps are saved with the SAME visual range used by the JET maps.
    # These files are for human inspection only, not metric calculation.
    pred_debug = ((pred_scalar_pano_m - depth_vmin_m) / (depth_vmax_m - depth_vmin_m + 1e-6)).clamp(0.0, 1.0)
    target_debug = ((target_scalar_pano_m - depth_vmin_m) / (depth_vmax_m - depth_vmin_m + 1e-6)).clamp(0.0, 1.0)
    torchvision.utils.save_image(pred_debug, os.path.join(args.output_dir, f"debug_scalar_pred_inverse_odi_epoch_{epoch:03d}.png"))
    torchvision.utils.save_image(target_debug, os.path.join(args.output_dir, f"debug_scalar_gt_inverse_odi_epoch_{epoch:03d}.png"))
    torchvision.utils.save_image(pred_valid_pano.float(), os.path.join(args.output_dir, f"debug_valid_inverse_odi_epoch_{epoch:03d}.png"))

    # Optional patch-grid output for debugging only. This uses the same visual
    # range as the inverse-ODI panorama, but it is only a patch-atlas diagnostic.
    if bool(getattr(args, "save_patch_grid_vis", False)):
        rgb_grid = _patches_to_grid(rgb_patches.float(), grid_h, grid_w)
        pred_grid_norm = ((pred_m - depth_vmin_m) / (depth_vmax_m - depth_vmin_m + 1e-6)).clamp(0.0, 1.0)
        target_grid_norm = ((target_m - depth_vmin_m) / (depth_vmax_m - depth_vmin_m + 1e-6)).clamp(0.0, 1.0)
        error_grid_norm = (error_m / (error_vmax_m + 1e-6)).clamp(0.0, 1.0)
        pred_grid = _patches_to_grid(_jet_colormap(pred_grid_norm.float(), valid_norm), grid_h, grid_w)
        target_grid = _patches_to_grid(_jet_colormap(target_grid_norm.float(), valid_norm), grid_h, grid_w)
        error_grid = _patches_to_grid(_jet_colormap(error_grid_norm.float(), valid_norm), grid_h, grid_w)
        sep_grid = sep[:, :, :rgb_grid.shape[-1]]
        grid_final = torch.cat([rgb_grid, sep_grid, pred_grid, sep_grid, target_grid, sep_grid, error_grid], dim=1)
        torchvision.utils.save_image(grid_final, os.path.join(args.output_dir, f"depth_patch_grid_compare_epoch_{epoch:03d}.png"))

    print(f"Saved inverse-ODI depth comparison to {compare_path}")


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args, epoch=0):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    max_depth = float(getattr(args, "max_depth", 100.0))
    all_pred = []
    all_target = []

    should_save_vis = (
        (epoch + 1 == args.epochs)
        or bool(getattr(args, "eval", False))
        or (getattr(args, "save_depth_vis_every", 0) > 0 and (epoch % args.save_depth_vis_every == 0))
    )
    saved_vis = False

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views, angles, targets, valid_mask = _unpack_depth_batch(batch)

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
            logits = model(views, angles)
            pred_resized, targets_resized, valid_resized = _resize_depth_pred_and_targets(logits, targets, args, valid_mask=valid_mask)
            depth_bin_logits = _resize_depth_bin_logits(_get_last_depth_bin_logits(model), args)
            depth_bin_edges_m = _get_depth_bin_edges_m(model)
            loss = criterion(
                pred_resized,
                targets_resized,
                valid_mask=valid_resized,
                depth_bin_logits=depth_bin_logits,
                depth_bin_edges_m=depth_bin_edges_m,
            )

        pred_meters = pred_resized.float().clamp(0.0, 1.0) * max_depth
        target_meters = targets_resized.float().clamp(0.0, 1.0) * max_depth

        valid = (target_meters > 1e-3) & (target_meters <= max_depth)
        if valid_resized is not None:
            valid = valid & (valid_resized.float() > 0.5)
        if valid.any():
            all_pred.append(pred_meters[valid].detach().cpu())
            all_target.append(target_meters[valid].detach().cpu())

        if should_save_vis and not saved_vis and batch_idx == 0:
            _save_depth_comparison(views, angles, pred_resized, targets_resized, valid_resized, args, epoch)
            saved_vis = True

        metric_logger.update(loss=loss.item())

    if len(all_pred) > 0:
        all_pred = torch.cat(all_pred)
        all_target = torch.cat(all_target)
        mae, rmse, abs_rel, delta1 = compute_depth_metrics(all_pred, all_target, max_depth=max_depth)
    else:
        mae, rmse, abs_rel, delta1 = 0.0, 0.0, 0.0, 0.0

    print(f"Validation RMSE: {rmse:.4f}m | AbsRel: {abs_rel:.4f} | Delta1: {delta1:.4f} | cliff={max_depth:g}m")

    return {
        "loss": metric_logger.loss.global_avg,
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta1": delta1,
    }
