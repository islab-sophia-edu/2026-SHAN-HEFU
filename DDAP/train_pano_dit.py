"""Stable Stage-2 PanoDiT training with paired GT validation.

Main safeguards compared with the earlier experimental trainer:
  * logit-normal base-time sampling followed by the latent-dimension shift;
  * conservative effective batch and learning-rate defaults;
  * linear warmup/decay schedule;
  * finite-but-catastrophic loss-ratio guard before backward;
  * paired validation from real noisy latents, with Pred/GT ERP images and
    tangent/ERP PSNR/SSIM;
  * unconditional samples remain separate because they have no paired GT.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import (
    EMA,
    append_jsonl,
    build_rae_from_pretrain,
    canonical_geometry,
    load_rae_checkpoint,
    save_checkpoint,
    save_panorama_batch,
    seed_everything,
    unpack_batch,
)
from flow_transport import DimensionShiftedRectifiedFlow
from latent_stats import LatentStandardizer
from pano_dit import PanoDiTDDT
from pano_metrics import compute_paired_metrics


OBJECTIVE = "standardized_dimension_shifted_rectified_flow_velocity_v3_stable_gt"


class RepeatFirstSample(Dataset):
    def __init__(self, dataset: Dataset, length: int) -> None:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        self.dataset = dataset
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        del index
        return self.dataset[0]


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return int(index), self.dataset[index]


def _source_names(loader: DataLoader, indices: torch.Tensor) -> list[str]:
    dataset = loader.dataset
    repeated = False
    if isinstance(dataset, IndexedDataset):
        dataset = dataset.dataset
    if isinstance(dataset, RepeatFirstSample):
        repeated = True
        dataset = dataset.dataset
    names = getattr(dataset, "image_files", None)
    result = []
    for raw in indices.detach().cpu().tolist():
        source_index = 0 if repeated else int(raw)
        if names is not None and 0 <= source_index < len(names):
            result.append(str(names[source_index]))
        else:
            result.append(f"index_{source_index}")
    return result


def _tensor_summary(tensor: torch.Tensor) -> dict:
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    result = {
        "shape": list(x.shape),
        "finite": bool(finite.all().item()),
        "finite_fraction": float(finite.float().mean().item()),
    }
    if finite.any():
        y = x[finite]
        result.update({
            "mean": float(y.mean().item()),
            "std": float(y.std(unbiased=False).item()),
            "abs_max": float(y.abs().max().item()),
            "min": float(y.min().item()),
            "max": float(y.max().item()),
        })
    return result


def _bad_gradients(model: torch.nn.Module, limit: int = 20) -> list[dict]:
    bad = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if not bool(finite.all().item()):
            finite_values = grad.detach()[finite]
            bad.append({
                "name": name,
                "shape": list(grad.shape),
                "finite_fraction": float(finite.float().mean().item()),
                "finite_abs_max": (
                    float(finite_values.abs().max().item())
                    if finite_values.numel() else None
                ),
            })
            if len(bad) >= limit:
                break
    return bad


def _dump_failure(
    output: Path,
    *,
    epoch: int,
    step: int,
    global_step: int,
    names: list[str],
    tensors: dict[str, torch.Tensor],
    reason: str,
    bad_gradients: Optional[list[dict]] = None,
) -> Path:
    folder = output / "instability_debug"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"epoch_{epoch:04d}_step_{step:07d}_update_{global_step:07d}"
    payload = {
        "reason": reason,
        "epoch": int(epoch),
        "step": int(step),
        "global_step": int(global_step),
        "files": names,
        "tensors": {name: _tensor_summary(value) for name, value in tensors.items()},
        "bad_gradients": bad_gradients or [],
    }
    json_path = folder / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {
            "files": names,
            **{name: value.detach().float().cpu() for name, value in tensors.items()},
        },
        folder / f"{stem}.pt",
    )
    print(f"Saved instability diagnostic: {json_path}")
    return json_path


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Stable PanoDiT training with paired GT validation")
    p.add_argument("--data_path", required=True)
    p.add_argument("--val_data_path", default="")
    p.add_argument("--pano_checkpoint", required=True)
    p.add_argument("--rae_checkpoint", required=True)
    p.add_argument("--latent_stats", required=True)
    p.add_argument("--output_dir", default="./output_pano_dit_v7")

    p.add_argument(
        "--pano_model",
        default="vit_huge_patch14",
        choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"],
    )
    p.add_argument("--pano_h", type=int, default=1024)
    p.add_argument("--pano_w", type=int, default=2048)
    p.add_argument("--grid_height", type=int, default=16)
    p.add_argument("--use_gctt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geometric_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_angular_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_gauge_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--angular_bias_init_slope", type=float, default=1.0)
    p.add_argument("--gauge_bias_init", type=float, default=0.0)
    p.add_argument("--allow_non_strict_pano_load", action="store_true")

    p.add_argument("--base_hidden_size", type=int, default=0)
    p.add_argument("--base_depth", type=int, default=12)
    p.add_argument("--base_num_heads", type=int, default=16)
    p.add_argument("--head_hidden_size", type=int, default=2048)
    p.add_argument("--head_depth", type=int, default=2)
    p.add_argument("--head_num_heads", type=int, default=16)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--head_use_geometry", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--base_flow_dimension", type=int, default=4096)
    p.add_argument("--time_eps", type=float, default=1e-3)
    p.add_argument(
        "--time_distribution",
        choices=["logit_normal", "uniform"],
        default="logit_normal",
    )
    p.add_argument("--logit_normal_mean", type=float, default=0.0)
    p.add_argument("--logit_normal_std", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--accum_iter",
        type=int,
        default=64,
        help="Default effective batch is 64 for batch_size=1",
    )
    p.add_argument("--lr", type=float, default=1.25e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--warmup_start_ratio", type=float, default=0.1)
    p.add_argument("--scheduler_type", choices=["linear", "cosine"], default="linear")
    p.add_argument("--decay_end_epoch", type=int, default=160)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--latitude_weighted_loss", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument(
        "--loss_guard_ratio",
        type=float,
        default=8.0,
        help="Reject a batch before backward when loss / zero-baseline exceeds this value; <=0 disables",
    )
    p.add_argument(
        "--loss_guard_absolute",
        type=float,
        default=16.0,
        help="Reject a batch before backward when velocity loss exceeds this value; <=0 disables",
    )
    p.add_argument("--max_skipped_batches_per_epoch", type=int, default=32)
    p.add_argument("--max_abs_standardized_latent", type=float, default=32.0)

    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--sample_count", type=int, default=1)
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--preview_weights", choices=["raw", "ema"], default="ema")
    p.add_argument("--save_every", type=int, default=5)

    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_samples", type=int, default=4)
    p.add_argument("--eval_save_samples", type=int, default=2)
    p.add_argument("--eval_base_times", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--eval_sampling_steps", type=int, default=50)
    p.add_argument("--eval_solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--metric_chunk_size", type=int, default=64)

    p.add_argument("--overfit_one", action="store_true")
    p.add_argument("--overfit_batches_per_epoch", type=int, default=200)

    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--aug_device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--detect_anomaly", action=argparse.BooleanOptionalAction, default=False)
    return p


def _make_panorama_dataset(args, root: str, *, training: bool) -> PanoramicDataset:
    # Stage-2 latent coordinates must remain fixed.  Geometry/color augmentation
    # is intentionally disabled for both train and paired validation.
    return PanoramicDataset(
        root_dir=root,
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


def make_train_dataset(args) -> Dataset:
    dataset: Dataset = _make_panorama_dataset(args, args.data_path, training=True)
    if args.overfit_one:
        dataset = RepeatFirstSample(dataset, args.overfit_batches_per_epoch * args.batch_size)
    return IndexedDataset(dataset)


def make_val_loader(args, device: torch.device) -> Optional[DataLoader]:
    if not args.val_data_path:
        return None
    dataset = _make_panorama_dataset(args, args.val_data_path, training=False)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _schedule_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_end_steps: int,
    min_ratio: float,
    warmup_start_ratio: float,
    schedule_type: str,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        progress = step / max(warmup_steps, 1)
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress

    denominator = max(decay_end_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / denominator
    progress = min(max(progress, 0.0), 1.0)
    if schedule_type == "linear":
        value = 1.0 - (1.0 - min_ratio) * progress
    else:
        value = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if step >= decay_end_steps:
        value = min_ratio
    return float(value)


def build_dit(args, latent_dim: int) -> PanoDiTDDT:
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


@torch.inference_mode()
def generate_unconditional_preview(
    rae,
    standardizer,
    model,
    transport,
    args,
    device,
    epoch: int,
) -> None:
    model.eval()
    angles, gauge, _ = canonical_geometry(args.grid_height, args.sample_count, device)
    shape = (args.sample_count, angles.shape[1], rae.latent_dim)
    generator = torch.Generator(device=device).manual_seed(args.seed + 123456)
    initial_noise = torch.randn(shape, generator=generator, device=device)
    x0_standardized = transport.sample(
        model,
        shape,
        angles,
        gauge if args.use_gctt else None,
        num_steps=args.sampling_steps,
        solver=args.solver,
        initial_noise=initial_noise,
    )
    z_decoder = standardizer.denormalize(x0_standardized.float())
    predicted_views = rae.decode(z_decoder, angles, gauge if args.use_gctt else None)
    save_panorama_batch(
        predicted_views.float(),
        angles,
        str(Path(args.output_dir) / "samples_unconditional"),
        f"epoch_{epoch + 1:04d}_{args.preview_weights}",
        args.pano_h,
        args.pano_w,
        args.grid_height,
        target_views=None,
        save_tangent_grid=False,
    )


def _add_metrics(total: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def _mean_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


@torch.inference_mode()
def evaluate_paired_gt(
    *,
    rae,
    standardizer,
    model,
    transport,
    val_loader: DataLoader,
    args,
    device: torch.device,
    epoch: int,
) -> dict:
    """Evaluate paired reconstruction/denoising; random generation has no GT."""
    model.eval()
    clean_total: dict[str, float] = {}
    noisy_totals = {float(t): {} for t in args.eval_base_times}
    processed = 0
    eval_dir = Path(args.output_dir) / "paired_gt" / f"epoch_{epoch + 1:04d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, batch in enumerate(val_loader):
        if processed >= args.eval_samples:
            break
        views, angles, gauge, _ = unpack_batch(batch)
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        gauge = gauge.to(device, non_blocking=True) if gauge is not None else None

        z_decoder = rae.encode(views, angles, gauge).float()
        x0 = standardizer.normalize(z_decoder)
        clean_views = rae.decode(z_decoder, angles, gauge)
        clean_metrics = compute_paired_metrics(
            clean_views.float(),
            views.float(),
            angles,
            args.pano_h,
            args.pano_w,
            args.grid_height,
            tangent_chunk_size=args.metric_chunk_size,
        )
        _add_metrics(clean_total, clean_metrics)

        if sample_index < args.eval_save_samples:
            save_panorama_batch(
                clean_views.float(),
                angles,
                str(eval_dir),
                f"sample_{sample_index:03d}_clean_decoder",
                args.pano_h,
                args.pano_w,
                args.grid_height,
                target_views=views.float(),
                save_tangent_grid=False,
            )

        generator = torch.Generator(device=device).manual_seed(
            args.seed + 700000 + sample_index
        )
        noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)

        for base_time in args.eval_base_times:
            base_time = float(base_time)
            if not args.time_eps <= base_time <= 1.0 - args.time_eps:
                raise ValueError(f"Invalid eval base time: {base_time}")
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
                num_steps=args.eval_sampling_steps,
                solver=args.eval_solver,
            )
            pred_views = rae.decode(standardizer.denormalize(x0_hat.float()), angles, gauge)
            metrics = compute_paired_metrics(
                pred_views.float(),
                views.float(),
                angles,
                args.pano_h,
                args.pano_w,
                args.grid_height,
                tangent_chunk_size=args.metric_chunk_size,
            )
            _add_metrics(noisy_totals[base_time], metrics)

            if sample_index < args.eval_save_samples:
                tag = f"t{base_time:.2f}".replace(".", "p")
                save_panorama_batch(
                    pred_views.float(),
                    angles,
                    str(eval_dir),
                    f"sample_{sample_index:03d}_{tag}",
                    args.pano_h,
                    args.pano_w,
                    args.grid_height,
                    target_views=views.float(),
                    save_tangent_grid=False,
                )
        processed += views.shape[0]

    report = {
        "epoch": int(epoch),
        "weights": args.preview_weights,
        "count": int(processed),
        "clean_decoder": _mean_metrics(clean_total, processed),
        "paired_denoising": {
            f"t{time_value:.2f}": _mean_metrics(total, processed)
            for time_value, total in noisy_totals.items()
        },
        "note": (
            "Paired GT metrics use real validation panoramas encoded to x0, "
            "corrupted at a fixed base time, denoised to t≈0, decoded, and "
            "inverse-ODI stitched. Unconditional random samples have no GT."
        ),
    }
    report_path = eval_dir / "metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"paired_validation": report}, ensure_ascii=False))
    return report


def main(args) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.aug_device == "cuda" and args.num_workers != 0:
        raise ValueError("Use --num_workers 0 with CUDA tangent extraction")
    if args.accum_iter < 1:
        raise ValueError("accum_iter must be >= 1")
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("Require 0 <= min_lr <= lr")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_dataset = make_train_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.overfit_one,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = make_val_loader(args, device)

    rae = build_rae_from_pretrain(args, device)
    rae_ckpt = load_rae_checkpoint(rae, args.rae_checkpoint, strict=True)
    if not bool(rae_ckpt.get("noise_robust", False)):
        print("WARNING: RAE decoder is not marked noise_robust. This does not affect Stage-2 loss, only decoding robustness.")
    rae.freeze_all()

    token_count = 2 * args.grid_height * args.grid_height
    standardizer = LatentStandardizer.load(args.latent_stats)
    standardizer.validate_shape(token_count, rae.latent_dim)
    print(
        f"Latent stats fingerprint={standardizer.fingerprint}, count={standardizer.count}, "
        f"shape=[1,{standardizer.token_count},{standardizer.latent_dim}]"
    )

    model = build_dit(args, rae.latent_dim).to(device)
    ema = EMA(model, decay=args.ema_decay)
    transport = DimensionShiftedRectifiedFlow(
        token_count,
        rae.latent_dim,
        base_dimension=args.base_flow_dimension,
        time_eps=args.time_eps,
        time_distribution=args.time_distribution,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
    )
    print(
        f"Flow N={token_count}, D={rae.latent_dim}, effective_dim={transport.effective_dimension}, "
        f"shift_alpha={transport.shift_alpha:.6f}, time_distribution={transport.time_distribution}"
    )
    trainable_m = sum(p.numel() for p in model.parameters()) / 1e6
    effective_batch = args.batch_size * args.accum_iter
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_iter)
    print(
        f"PanoDiT trainable={trainable_m:.2f}M, effective_batch={effective_batch}, "
        f"optimizer_updates_per_epoch≈{updates_per_epoch}, lr={args.lr:.8g}"
    )
    if effective_batch < 32:
        print("WARNING: effective_batch < 32 is high-variance for this 527M-class model.")
    if updates_per_epoch > 5000:
        print("WARNING: updates_per_epoch > 5000; increase --accum_iter before full training.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_steps = max(args.epochs * updates_per_epoch, 1)
    warmup_steps = args.warmup_epochs * updates_per_epoch
    decay_end_steps = min(args.decay_end_epoch, args.epochs) * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _schedule_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_end_steps=decay_end_steps,
            min_ratio=args.min_lr / args.lr,
            warmup_start_ratio=args.warmup_start_ratio,
            schedule_type=args.scheduler_type,
        ),
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch.autograd.set_detect_anomaly(args.detect_anomaly, check_nan=True)

    start_epoch = 0
    global_step = 0
    best_val_ssim = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("objective") != OBJECTIVE:
            raise RuntimeError("Do not resume old unstable/v4 checkpoints into v7")
        if checkpoint.get("latent_stats_fingerprint") != standardizer.fingerprint:
            raise RuntimeError("Resume checkpoint used different latent statistics")
        model.load_state_dict(checkpoint["dit"], strict=True)
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_ssim = float(checkpoint.get("best_val_ssim", best_val_ssim))

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        running_loss = 0.0
        running_zero = 0.0
        running_z_mean = 0.0
        running_z_std = 0.0
        accepted_samples = 0
        skipped_batches = 0
        last_context = None

        for step, batch in enumerate(train_loader):
            indices, sample_batch = batch
            names = _source_names(train_loader, indices)
            views, angles, gauge, weights = unpack_batch(sample_batch)
            views = views.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            gauge = gauge.to(device, non_blocking=True) if gauge is not None else None
            weights = weights.to(device, non_blocking=True) if args.latitude_weighted_loss else None

            with torch.inference_mode():
                z_decoder = rae.encode(views, angles, gauge).float()
                z0 = standardizer.normalize(z_decoder)
            flow_batch = transport.make_training_batch(z0)

            pre_tensors = {
                "views": views,
                "angles": angles,
                "z_decoder": z_decoder,
                "z0": z0,
                "x_t": flow_batch.x_t,
                "target_velocity": flow_batch.target_velocity,
                "time": flow_batch.time,
                "base_time": flow_batch.base_time,
            }
            latent_abs_max = float(z0.detach().abs().max().item())
            pre_finite = all(torch.isfinite(value).all() for value in pre_tensors.values())
            latent_outlier = (
                args.max_abs_standardized_latent > 0
                and latent_abs_max > args.max_abs_standardized_latent
            )
            if not pre_finite or latent_outlier:
                reason = "nonfinite_input" if not pre_finite else f"z0_abs_max={latent_abs_max:.6f}"
                _dump_failure(
                    output,
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    names=names,
                    tensors=pre_tensors,
                    reason=reason,
                )
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                skipped_batches += 1
                if skipped_batches > args.max_skipped_batches_per_epoch:
                    raise FloatingPointError("Too many invalid input batches")
                continue

            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted_velocity = model(flow_batch.x_t, flow_batch.time, angles, gauge)
                loss = transport.velocity_loss(
                    predicted_velocity, flow_batch.target_velocity, weights
                )
            zero_baseline = flow_batch.target_velocity.float().square().mean()
            loss_ratio = loss.detach().float() / zero_baseline.detach().float().clamp_min(1e-12)

            forward_bad = (
                not torch.isfinite(predicted_velocity).all()
                or not torch.isfinite(loss)
                or not torch.isfinite(loss_ratio)
            )
            guard_bad = (
                (args.loss_guard_ratio > 0 and float(loss_ratio) > args.loss_guard_ratio)
                or (args.loss_guard_absolute > 0 and float(loss.detach()) > args.loss_guard_absolute)
            )
            if forward_bad or guard_bad:
                reason = (
                    "nonfinite_forward"
                    if forward_bad
                    else f"finite_loss_spike_loss={float(loss.detach()):.6f}_ratio={float(loss_ratio):.6f}"
                )
                _dump_failure(
                    output,
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    names=names,
                    tensors={
                        **pre_tensors,
                        "predicted_velocity": predicted_velocity,
                        "loss": loss,
                        "zero_baseline": zero_baseline,
                        "loss_ratio": loss_ratio,
                    },
                    reason=reason,
                )
                # Drop the whole partially accumulated update.  Keeping earlier
                # gradients and silently replacing one member changes the batch.
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                skipped_batches += 1
                if skipped_batches > args.max_skipped_batches_per_epoch:
                    raise FloatingPointError(
                        "Too many finite/non-finite loss spikes; stop and inspect instability_debug"
                    )
                continue

            scaler.scale(loss / args.accum_iter).backward()
            accum_count += 1
            last_context = (step, names, pre_tensors, predicted_velocity, loss)

            with torch.no_grad():
                b = views.shape[0]
                running_loss += float(loss.detach()) * b
                running_zero += float(zero_baseline) * b
                running_z_mean += float(z0.mean()) * b
                running_z_std += float(z0.std(unbiased=False)) * b
                accepted_samples += b

            is_last = step + 1 == len(train_loader)
            if accum_count >= args.accum_iter or (is_last and accum_count > 0):
                scaler.unscale_(optimizer)
                bad_gradients = _bad_gradients(model)
                if bad_gradients:
                    context_step, context_names, context_tensors, context_pred, context_loss = last_context
                    _dump_failure(
                        output,
                        epoch=epoch,
                        step=context_step,
                        global_step=global_step,
                        names=context_names,
                        tensors={
                            **context_tensors,
                            "predicted_velocity": context_pred,
                            "loss": context_loss,
                        },
                        reason="nonfinite_parameter_gradient",
                        bad_gradients=bad_gradients,
                    )
                    raise FloatingPointError(
                        f"Non-finite parameter gradient; first={bad_gradients[0]['name']}"
                    )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1
                accum_count = 0

                if args.log_every > 0 and global_step % args.log_every == 0:
                    print({
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": float(loss.detach()),
                        "loss_to_zero": float(loss_ratio),
                        "grad_norm_before_clip": float(grad_norm.detach()),
                        "z0_abs_max": latent_abs_max,
                        "base_time": float(flow_batch.base_time.mean()),
                        "shifted_time": float(flow_batch.time.mean()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "skipped_batches": skipped_batches,
                        "files": names,
                    })

        if accepted_samples == 0:
            raise RuntimeError("No accepted samples in epoch")

        row = {
            "epoch": epoch,
            "velocity_loss": running_loss / accepted_samples,
            "zero_predictor_loss": running_zero / accepted_samples,
            "standardized_z_mean": running_z_mean / accepted_samples,
            "standardized_z_std": running_z_std / accepted_samples,
            "loss_to_zero_ratio": running_loss / max(running_zero, 1e-12),
            "accepted_samples": accepted_samples,
            "skipped_batches": skipped_batches,
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "effective_batch": effective_batch,
            "updates_per_epoch": updates_per_epoch,
            "time_distribution": transport.time_distribution,
            "shift_alpha": transport.shift_alpha,
            "stats_fingerprint": standardizer.fingerprint,
            "seconds": time.time() - started,
        }
        print(row)
        append_jsonl(str(output / "log.jsonl"), row)

        preview_model = model if args.preview_weights == "raw" else ema.model
        val_report = None
        if val_loader is not None and args.eval_every > 0 and (
            (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        ):
            val_report = evaluate_paired_gt(
                rae=rae,
                standardizer=standardizer,
                model=preview_model,
                transport=transport,
                val_loader=val_loader,
                args=args,
                device=device,
                epoch=epoch,
            )
            key = "t0.50" if "t0.50" in val_report["paired_denoising"] else next(
                iter(val_report["paired_denoising"])
            )
            current_ssim = float(val_report["paired_denoising"][key]["erp_ssim"])
            best_val_ssim = max(best_val_ssim, current_ssim)

        payload = {
            "dit": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_ssim": best_val_ssim,
            "args": vars(args),
            "latent_dim": rae.latent_dim,
            "token_count": token_count,
            "objective": OBJECTIVE,
            "shift_alpha": transport.shift_alpha,
            "latent_stats": standardizer.checkpoint_payload(),
            "latent_stats_fingerprint": standardizer.fingerprint,
            "paired_validation": val_report,
        }
        save_checkpoint(str(output / "checkpoint_latest.pt"), payload)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(str(output / f"checkpoint_epoch_{epoch + 1:04d}.pt"), payload)

        if val_report is not None:
            key = "t0.50" if "t0.50" in val_report["paired_denoising"] else next(
                iter(val_report["paired_denoising"])
            )
            current_ssim = float(val_report["paired_denoising"][key]["erp_ssim"])
            if current_ssim >= best_val_ssim - 1e-12:
                save_checkpoint(str(output / "checkpoint_best_paired.pt"), payload)

        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            generate_unconditional_preview(
                rae,
                standardizer,
                preview_model,
                transport,
                args,
                device,
                epoch,
            )


if __name__ == "__main__":
    main(get_parser().parse_args())
