"""Paired image-quality metrics for PanoRAE/PanoDiT.

Metrics are reported in two domains:
  * tangent domain: compares the decoded ODI/tangent patches directly;
  * ERP domain: inverse-projects prediction and target with the same geometry,
    then compares the resulting panoramas.

PSNR/SSIM are only meaningful for paired prediction/target data.  Do not use
these metrics for unconditional random samples unless the model was explicitly
trained to overfit the same single target image.
"""
from __future__ import annotations

from typing import Dict

import torch

from common import denormalize_views, views_to_erp

try:
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
except ImportError as exc:  # pragma: no cover - explicit runtime guidance
    raise ImportError(
        "pano_metrics.py requires torchmetrics. Install with: pip install torchmetrics"
    ) from exc


def _check_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if pred.ndim != 5 or pred.shape[2] != 3:
        raise ValueError(
            f"Expected normalized tangent views [B,N,3,H,W], got {tuple(pred.shape)}"
        )
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Prediction or target contains NaN/Inf")


def _mean_patch_psnr_ssim(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[float, float]:
    """Average per-patch PSNR/SSIM without materializing all metric buffers."""
    b, n, c, h, w = pred_rgb.shape
    pred_flat = pred_rgb.reshape(b * n, c, h, w)
    target_flat = target_rgb.reshape(b * n, c, h, w)

    psnr_sum = torch.zeros((), device=pred_rgb.device, dtype=torch.float64)
    ssim_sum = torch.zeros((), device=pred_rgb.device, dtype=torch.float64)
    count = 0

    for start in range(0, pred_flat.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), pred_flat.shape[0])
        p = pred_flat[start:end].float().clamp(0.0, 1.0)
        t = target_flat[start:end].float().clamp(0.0, 1.0)

        psnr_values = peak_signal_noise_ratio(
            p,
            t,
            data_range=1.0,
            reduction="none",
            dim=(1, 2, 3),
        )
        ssim_values = structural_similarity_index_measure(
            p,
            t,
            data_range=1.0,
            reduction="none",
        )
        psnr_sum += psnr_values.double().sum()
        ssim_sum += ssim_values.double().sum()
        count += int(end - start)

    return float(psnr_sum / max(count, 1)), float(ssim_sum / max(count, 1))


def _erp_pair(
    pred_views: torch.Tensor,
    target_views: torch.Tensor,
    angles: torch.Tensor,
    pano_h: int,
    pano_w: int,
    grid_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_erps = []
    target_erps = []
    for index in range(pred_views.shape[0]):
        pred_erps.append(
            views_to_erp(
                pred_views[index],
                angles[index],
                pano_h,
                pano_w,
                grid_height,
            )
        )
        target_erps.append(
            views_to_erp(
                target_views[index],
                angles[index],
                pano_h,
                pano_w,
                grid_height,
            )
        )
    return torch.stack(pred_erps, dim=0), torch.stack(target_erps, dim=0)


def _weighted_spherical_psnr(
    pred_erp: torch.Tensor,
    target_erp: torch.Tensor,
) -> float:
    """Cosine-latitude weighted ERP PSNR.

    ERP rows oversample polar regions.  This metric weights each row by the
    spherical area element cos(latitude), while retaining a 1.0 RGB range.
    """
    if pred_erp.shape != target_erp.shape or pred_erp.ndim != 4:
        raise ValueError("ERP tensors must share [B,3,H,W] shape")
    h = pred_erp.shape[-2]
    # Pixel-center latitude: north (+pi/2) to south (-pi/2).
    rows = torch.arange(h, device=pred_erp.device, dtype=torch.float32)
    latitude = torch.pi / 2.0 - (rows + 0.5) * torch.pi / h
    weights = torch.cos(latitude).clamp_min(0.0).view(1, 1, h, 1)
    squared = (pred_erp.float() - target_erp.float()).square()
    weighted_mse = (squared * weights).sum() / (
        weights.sum() * pred_erp.shape[0] * pred_erp.shape[1] * pred_erp.shape[-1]
    ).clamp_min(1e-12)
    psnr = 10.0 * torch.log10(1.0 / weighted_mse.clamp_min(1e-12))
    return float(psnr)


@torch.no_grad()
def compute_paired_metrics(
    predicted_views: torch.Tensor,
    target_views: torch.Tensor,
    angles: torch.Tensor,
    pano_h: int,
    pano_w: int,
    grid_height: int,
    tangent_chunk_size: int = 64,
) -> Dict[str, float]:
    """Compute tangent and inverse-ODI ERP metrics for paired outputs."""
    _check_pair(predicted_views, target_views)
    if angles.shape[:2] != predicted_views.shape[:2]:
        raise ValueError("angles and tangent views have incompatible token counts")

    pred_rgb = denormalize_views(predicted_views).float().clamp(0.0, 1.0)
    target_rgb = denormalize_views(target_views).float().clamp(0.0, 1.0)

    tangent_psnr, tangent_ssim = _mean_patch_psnr_ssim(
        pred_rgb,
        target_rgb,
        chunk_size=tangent_chunk_size,
    )

    pred_erp, target_erp = _erp_pair(
        predicted_views,
        target_views,
        angles,
        pano_h,
        pano_w,
        grid_height,
    )
    pred_erp = pred_erp.float().clamp(0.0, 1.0)
    target_erp = target_erp.float().clamp(0.0, 1.0)

    erp_psnr = peak_signal_noise_ratio(
        pred_erp,
        target_erp,
        data_range=1.0,
        reduction="elementwise_mean",
    )
    erp_ssim = structural_similarity_index_measure(
        pred_erp,
        target_erp,
        data_range=1.0,
        reduction="elementwise_mean",
    )

    return {
        "tangent_psnr": tangent_psnr,
        "tangent_ssim": tangent_ssim,
        "erp_psnr": float(erp_psnr),
        "erp_ssim": float(erp_ssim),
        "erp_ws_psnr": _weighted_spherical_psnr(pred_erp, target_erp),
    }
