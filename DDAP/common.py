"""Shared utilities for the corrected PanoRAE/PanoDiT pipeline."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision

from models_PanoMAE_gctt import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from pano_rae import PanoRepresentationAutoencoder, load_pretrained_pano_mae


MODEL_FACTORY = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch):
    if len(batch) == 4:
        views, angles, gauge_angles, weights = batch
    elif len(batch) == 3:
        views, angles, weights = batch
        gauge_angles = None
    else:
        raise ValueError(f"Expected batch length 3 or 4, received {len(batch)}")
    return views, angles, gauge_angles, weights


def build_pano_mae(args) -> torch.nn.Module:
    if args.pano_model not in MODEL_FACTORY:
        raise ValueError(f"Unknown PanoMAE model: {args.pano_model}")
    if args.pano_h % args.grid_height != 0:
        raise ValueError("pano_h must be divisible by grid_height")
    patch_size = args.pano_h // args.grid_height
    return MODEL_FACTORY[args.pano_model](
        img_size=patch_size,
        norm_pix_loss=False,
        geometric_bias=args.geometric_bias,
        adaptive_masking=False,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        use_gctt=args.use_gctt,
        use_gauge_bias=args.use_gauge_bias,
        gauge_bias_init=args.gauge_bias_init,
    )


def build_rae_from_pretrain(args, device: torch.device) -> PanoRepresentationAutoencoder:
    pano_mae = build_pano_mae(args)
    load_pretrained_pano_mae(
        pano_mae,
        args.pano_checkpoint,
        strict=not args.allow_non_strict_pano_load,
    )
    return PanoRepresentationAutoencoder(pano_mae, normalize_latents=True).to(device)


def load_rae_checkpoint(
    rae: PanoRepresentationAutoencoder,
    path: str,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("rae", checkpoint.get("model", checkpoint))
    incompatible = rae.load_state_dict(state, strict=strict)
    if not strict:
        print("RAE missing keys:", incompatible.missing_keys)
        print("RAE unexpected keys:", incompatible.unexpected_keys)
    return checkpoint


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        target = self.model.state_dict()
        for key, value in target.items():
            src = source[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state) -> None:
        self.model.load_state_dict(state)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_geometry(
    grid_height: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_num = int(grid_height)
    u_num = 2 * v_num
    v_fov = 180.0 / v_num
    phis = np.linspace(90.0 - v_fov / 2.0, -90.0 + v_fov / 2.0, v_num)
    thetas = np.linspace(-180.0, 180.0, u_num, endpoint=False)
    grid_phi, grid_theta = np.meshgrid(phis, thetas, indexing="ij")
    angles_np = np.stack([grid_theta.reshape(-1), grid_phi.reshape(-1)], axis=-1)
    angles = torch.tensor(angles_np, dtype=torch.float32, device=device)
    angles = angles.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    gauge = torch.zeros(batch_size, angles.shape[1], 1, device=device)
    weights = torch.cos(torch.deg2rad(angles[..., 1]))
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return angles, gauge, weights


def denormalize_views(views: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=views.device, dtype=views.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=views.device, dtype=views.dtype)
    shape = [1] * views.ndim
    shape[-3] = 3
    return (views * std.view(*shape) + mean.view(*shape)).clamp(0.0, 1.0)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        finite_ratio = torch.isfinite(tensor).float().mean().item()
        raise FloatingPointError(f"{name} contains non-finite values; finite_ratio={finite_ratio:.6f}")


@torch.no_grad()
def views_to_erp(
    normalized_views: torch.Tensor,
    angles: torch.Tensor,
    pano_h: int,
    pano_w: int,
    grid_height: int,
) -> torch.Tensor:
    """Inverse-ODI stitch one sample of tangent views into an ERP panorama."""
    from util import odi_processing as odi

    if normalized_views.ndim != 4:
        raise ValueError(f"Expected [N,3,H,W], got {tuple(normalized_views.shape)}")
    rgb = denormalize_views(normalized_views)
    _assert_finite("RGB tangent views", rgb)
    u_steps = 2 * int(grid_height)
    h_fov = 360.0 / u_steps
    v_fov = 180.0 / int(grid_height)
    panorama = odi.embed_patches_to_pano_gpu(
        rgb,
        int(pano_h),
        int(pano_w),
        u_steps,
        int(grid_height),
        h_fov,
        v_fov,
        angles,
    )
    _assert_finite("ERP panorama", panorama)
    return panorama.clamp(0.0, 1.0)


@torch.no_grad()
def save_panorama_batch(
    predicted_views: torch.Tensor,
    angles: torch.Tensor,
    output_dir: str,
    prefix: str,
    pano_h: int,
    pano_w: int,
    grid_height: int,
    target_views: Optional[torch.Tensor] = None,
    save_tangent_grid: bool = False,
) -> None:
    """Save inverse-ODI ERP output.

    ``save_tangent_grid`` is diagnostic-only and defaults to False.  Therefore
    normal training/sampling emits only ERP images, not a second grid output.
    """
    from util import odi_processing as odi

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _assert_finite("predicted_views", predicted_views)

    pred_rgb = denormalize_views(predicted_views)
    target_rgb = denormalize_views(target_views) if target_views is not None else None
    u_steps = 2 * int(grid_height)

    for index in range(pred_rgb.shape[0]):
        pred_erp = views_to_erp(
            predicted_views[index], angles[index], pano_h, pano_w, grid_height
        )
        torchvision.utils.save_image(
            pred_erp,
            output / f"{prefix}_{index:03d}_pred_erp.png",
        )

        if target_views is not None:
            gt_erp = views_to_erp(
                target_views[index], angles[index], pano_h, pano_w, grid_height
            )
            torchvision.utils.save_image(
                gt_erp,
                output / f"{prefix}_{index:03d}_gt_erp.png",
            )
            separator = torch.ones(
                pred_erp.shape[0], 4, pred_erp.shape[-1],
                device=pred_erp.device, dtype=pred_erp.dtype,
            )
            comparison = torch.cat([gt_erp, separator, pred_erp], dim=1)
            torchvision.utils.save_image(
                comparison,
                output / f"{prefix}_{index:03d}_gt_top_pred_bottom.png",
            )

        if save_tangent_grid:
            grid = odi.make_grid_gpu(pred_rgb[index], grid_width=u_steps)
            torchvision.utils.save_image(
                grid,
                output / f"{prefix}_{index:03d}_diagnostic_tangent_grid.png",
            )
            if target_rgb is not None:
                gt_grid = odi.make_grid_gpu(target_rgb[index], grid_width=u_steps)
                torchvision.utils.save_image(
                    gt_grid,
                    output / f"{prefix}_{index:03d}_diagnostic_gt_tangent_grid.png",
                )
