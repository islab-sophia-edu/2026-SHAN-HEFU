"""Paired GT evaluation for PanoDiT checkpoints.

This script is intentionally different from unconditional sampling.  For every
validation panorama it encodes the real image to x0, corrupts x0 at fixed base
times, integrates the learned flow back to t≈0, decodes, inverse-ODI stitches,
and computes paired PSNR/SSIM.  It also saves GT-top/Pred-bottom comparisons.
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import (
    build_rae_from_pretrain,
    load_rae_checkpoint,
    save_panorama_batch,
    seed_everything,
    unpack_batch,
)
from flow_transport import DimensionShiftedRectifiedFlow
from latent_stats import LatentStandardizer
from pano_dit import PanoDiTDDT
from pano_metrics import compute_paired_metrics


ACCEPTED_OBJECTIVES = {
    "standardized_dimension_shifted_rectified_flow_velocity_v2",
    "standardized_dimension_shifted_rectified_flow_velocity_v3_stable_gt",
}


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate PanoDiT with paired GT panoramas")
    p.add_argument("--data_path", required=True)
    p.add_argument("--dit_checkpoint", required=True)
    p.add_argument("--pano_checkpoint", required=True)
    p.add_argument("--rae_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--weights", choices=["raw", "ema"], default="ema")
    p.add_argument("--base_times", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--max_samples", type=int, default=16, help="0 evaluates all")
    p.add_argument("--save_samples", type=int, default=4)
    p.add_argument("--metric_chunk_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aug_device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_non_strict_pano_load", action="store_true")
    return p


def _restore_args(cli, saved: dict) -> Namespace:
    args = Namespace(**saved)
    args.pano_checkpoint = cli.pano_checkpoint
    args.rae_checkpoint = cli.rae_checkpoint
    args.allow_non_strict_pano_load = cli.allow_non_strict_pano_load
    args.device = cli.device
    args.aug_device = cli.aug_device
    args.num_workers = cli.num_workers
    # Backward-compatible fallbacks for old v4 checkpoints.
    args.time_distribution = getattr(args, "time_distribution", "uniform")
    args.logit_normal_mean = getattr(args, "logit_normal_mean", 0.0)
    args.logit_normal_std = getattr(args, "logit_normal_std", 1.0)
    return args


def _build_model(args, latent_dim: int) -> PanoDiTDDT:
    base_width = latent_dim if args.base_hidden_size == 0 else args.base_hidden_size
    return PanoDiTDDT(
        latent_dim=latent_dim,
        base_hidden_size=base_width,
        base_depth=args.base_depth,
        base_num_heads=args.base_num_heads,
        head_hidden_size=args.head_hidden_size,
        head_depth=args.head_depth,
        head_num_heads=args.head_num_heads,
        mlp_ratio=args.mlp_ratio,
        geometric_bias=args.geometric_bias,
        use_gctt=args.use_gctt,
        use_angular_bias=args.use_angular_bias,
        use_gauge_bias=args.use_gauge_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        gauge_bias_init=args.gauge_bias_init,
        dropout=args.dropout,
        head_use_geometry=args.head_use_geometry,
    )


def _dataset(args, path: str) -> PanoramicDataset:
    return PanoramicDataset(
        root_dir=path,
        grid_height=args.grid_height,
        img_size=args.pano_h // args.grid_height,
        pano_h=args.pano_h,
        pano_w=args.pano_w,
        multiscale_sampling=False,
        use_full_pose3d=False,
        use_color_jitter=False,
        use_blur=False,
        use_horizontal_roll=False,
        is_training=False,
        angle_jitter_deg=0.0,
        aug_device=args.aug_device,
        use_gctt=args.use_gctt,
        gctt_gauge_jitter_deg=0.0,
        gctt_local_gauge_jitter_deg=0.0,
    )


def _add(total: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value)


def _mean(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def main(cli) -> None:
    if cli.aug_device == "cuda" and cli.num_workers != 0:
        raise ValueError("Use --num_workers 0 with --aug_device cuda")
    seed_everything(cli.seed)
    device = torch.device(cli.device)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(cli.dit_checkpoint, map_location="cpu", weights_only=False)
    objective = checkpoint.get("objective")
    if objective not in ACCEPTED_OBJECTIVES:
        raise RuntimeError(f"Unsupported Stage-2 objective: {objective}")
    args = _restore_args(cli, checkpoint["args"])

    standardizer = LatentStandardizer.from_payload(checkpoint["latent_stats"])
    if standardizer.fingerprint != checkpoint.get("latent_stats_fingerprint"):
        raise RuntimeError("Checkpoint latent stats fingerprint mismatch")

    rae = build_rae_from_pretrain(args, device)
    load_rae_checkpoint(rae, cli.rae_checkpoint, strict=True)
    rae.freeze_all()
    token_count = 2 * args.grid_height * args.grid_height
    standardizer.validate_shape(token_count, rae.latent_dim)

    model = _build_model(args, rae.latent_dim).to(device)
    key = "ema" if cli.weights == "ema" and "ema" in checkpoint else "dit"
    model.load_state_dict(checkpoint[key], strict=True)
    model.eval()

    transport = DimensionShiftedRectifiedFlow(
        token_count,
        rae.latent_dim,
        base_dimension=args.base_flow_dimension,
        time_eps=args.time_eps,
        time_distribution=args.time_distribution,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
    )
    loader = DataLoader(
        _dataset(args, cli.data_path),
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    clean_total: dict[str, float] = {}
    noisy_totals = {float(t): {} for t in cli.base_times}
    count = 0

    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            if cli.max_samples > 0 and count >= cli.max_samples:
                break
            views, angles, gauge, _ = unpack_batch(batch)
            views = views.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            gauge = gauge.to(device, non_blocking=True) if gauge is not None else None

            z_decoder = rae.encode(views, angles, gauge).float()
            x0 = standardizer.normalize(z_decoder)
            clean_pred = rae.decode(z_decoder, angles, gauge)
            clean_metrics = compute_paired_metrics(
                clean_pred.float(), views.float(), angles,
                args.pano_h, args.pano_w, args.grid_height,
                tangent_chunk_size=cli.metric_chunk_size,
            )
            _add(clean_total, clean_metrics)

            if sample_index < cli.save_samples:
                save_panorama_batch(
                    clean_pred.float(), angles, str(output),
                    f"sample_{sample_index:03d}_clean_decoder",
                    args.pano_h, args.pano_w, args.grid_height,
                    target_views=views.float(), save_tangent_grid=False,
                )

            generator = torch.Generator(device=device).manual_seed(
                cli.seed + 900000 + sample_index
            )
            noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)

            for base_time in cli.base_times:
                base_time = float(base_time)
                shifted = transport.shift_time(
                    torch.tensor([base_time], device=device, dtype=x0.dtype)
                )
                x_t = (1.0 - shifted.view(1, 1, 1)) * x0 + shifted.view(1, 1, 1) * noise
                x0_hat = transport.denoise_from_base_time(
                    model,
                    x_t,
                    angles,
                    gauge if args.use_gctt else None,
                    base_time=base_time,
                    num_steps=cli.sampling_steps,
                    solver=cli.solver,
                )
                pred_views = rae.decode(
                    standardizer.denormalize(x0_hat.float()), angles, gauge
                )
                metrics = compute_paired_metrics(
                    pred_views.float(), views.float(), angles,
                    args.pano_h, args.pano_w, args.grid_height,
                    tangent_chunk_size=cli.metric_chunk_size,
                )
                _add(noisy_totals[base_time], metrics)

                if sample_index < cli.save_samples:
                    tag = f"t{base_time:.2f}".replace(".", "p")
                    save_panorama_batch(
                        pred_views.float(), angles, str(output),
                        f"sample_{sample_index:03d}_{tag}",
                        args.pano_h, args.pano_w, args.grid_height,
                        target_views=views.float(), save_tangent_grid=False,
                    )
            count += 1

    report = {
        "checkpoint": cli.dit_checkpoint,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "objective": objective,
        "weights": key,
        "count": count,
        "stats_fingerprint": standardizer.fingerprint,
        "clean_decoder": _mean(clean_total, count),
        "paired_denoising": {
            f"t{time_value:.2f}": _mean(total, count)
            for time_value, total in noisy_totals.items()
        },
        "note": (
            "These are paired denoising metrics. An unconditional sample from "
            "pure random noise has no corresponding GT and must not be assigned PSNR/SSIM."
        ),
    }
    (output / "paired_gt_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved Pred/GT ERP comparisons to {output.resolve()}")


if __name__ == "__main__":
    main(get_parser().parse_args())
