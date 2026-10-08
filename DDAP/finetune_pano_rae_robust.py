"""Fine-tune an existing clean PanoRAE decoder with stochastic latent noise."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import (
    append_jsonl,
    build_rae_from_pretrain,
    load_rae_checkpoint,
    save_checkpoint,
    save_panorama_batch,
    seed_everything,
    unpack_batch,
)
from pano_rae import latitude_weighted_mse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Noise-robust PanoRAE decoder fine-tuning")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_data_path", default="")
    parser.add_argument("--pano_checkpoint", required=True)
    parser.add_argument("--init_rae_checkpoint", required=True,
                        help="The existing clean Stage-1 checkpoint_best.pt")
    parser.add_argument("--output_dir", default="./output_pano_rae_robust")

    parser.add_argument("--pano_model", default="vit_base_patch16",
                        choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"])
    parser.add_argument("--pano_h", type=int, default=512)
    parser.add_argument("--pano_w", type=int, default=1024)
    parser.add_argument("--grid_height", type=int, default=16)
    parser.add_argument("--use_gctt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geometric_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_angular_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_gauge_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--angular_bias_init_slope", type=float, default=1.0)
    parser.add_argument("--gauge_bias_init", type=float, default=0.0)
    parser.add_argument("--allow_non_strict_pano_load", action="store_true")

    parser.add_argument("--noise_tau", type=float, default=0.8,
                        help="sigma ~ |N(0,tau^2)|, then z_noisy=z+sigma*epsilon")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--latitude_weighted_loss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aug_device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--preview_every", type=int, default=5)
    return parser


def make_dataset(path: str, args) -> PanoramicDataset:
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


def lr_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(step + 1, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def add_stochastic_latent_noise(z: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    if tau <= 0:
        sigma = torch.zeros(z.shape[0], 1, 1, device=z.device, dtype=z.dtype)
        return z, sigma
    sigma = torch.randn(z.shape[0], 1, 1, device=z.device, dtype=z.dtype).abs() * float(tau)
    return z + sigma * torch.randn_like(z), sigma


@torch.no_grad()
def evaluate(rae, loader, device, use_amp: bool, weighted: bool, tau: float) -> tuple[float, float]:
    rae.eval()
    clean_total = robust_total = 0.0
    samples = 0
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(12345)
    for batch in loader:
        views, angles, gauge, weights = unpack_batch(batch)
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        gauge = gauge.to(device, non_blocking=True) if gauge is not None else None
        weights = weights.to(device, non_blocking=True) if weighted else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            z = rae.encode(views, angles, gauge)
            clean_pred = rae.decode(z, angles, gauge)
            noisy_z, _ = add_stochastic_latent_noise(z, tau)
            robust_pred = rae.decode(noisy_z, angles, gauge)
            clean_loss = latitude_weighted_mse(clean_pred, views, weights)
            robust_loss = latitude_weighted_mse(robust_pred, views, weights)
        batch_size = views.shape[0]
        clean_total += float(clean_loss) * batch_size
        robust_total += float(robust_loss) * batch_size
        samples += batch_size
    torch.random.set_rng_state(generator_state)
    return clean_total / max(samples, 1), robust_total / max(samples, 1)


def main(args) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.aug_device == "cuda" and args.num_workers != 0:
        raise ValueError("Use --num_workers 0 with CUDA patch extraction")
    if args.noise_tau <= 0:
        raise ValueError("noise_tau must be > 0 for robust fine-tuning")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(args.data_path, args)
    val_set = make_dataset(args.val_data_path, args) if args.val_data_path else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    ) if val_set is not None else None

    rae = build_rae_from_pretrain(args, device)
    load_rae_checkpoint(rae, args.init_rae_checkpoint, strict=True)
    rae.freeze_encoder()
    trainable = [parameter for parameter in rae.parameters() if parameter.requires_grad]
    print(f"Loaded clean RAE: {args.init_rae_checkpoint}")
    print(f"Noise robust fine-tune: sigma ~ |N(0,{args.noise_tau}^2)|")
    print(f"Trainable decoder parameters: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_iter)
    total_steps = max(args.epochs * updates_per_epoch, 1)
    warmup_steps = args.warmup_epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, total_steps, warmup_steps, args.min_lr / args.lr),
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_robust = float("inf")

    for epoch in range(args.epochs):
        started = time.time()
        rae.train(True)
        optimizer.zero_grad(set_to_none=True)
        running = sigma_running = 0.0
        samples = 0

        for step, batch in enumerate(train_loader):
            views, angles, gauge, weights = unpack_batch(batch)
            views = views.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            gauge = gauge.to(device, non_blocking=True) if gauge is not None else None
            weights = weights.to(device, non_blocking=True) if args.latitude_weighted_loss else None

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                z = rae.encode(views, angles, gauge)
            noisy_z, sigma = add_stochastic_latent_noise(z, args.noise_tau)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = rae.decode(noisy_z, angles, gauge)
                loss = latitude_weighted_mse(pred, views, weights)
                scaled_loss = loss / args.accum_iter

            scaler.scale(scaled_loss).backward()
            update = ((step + 1) % args.accum_iter == 0) or (step + 1 == len(train_loader))
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            batch_size = views.shape[0]
            running += float(loss.detach()) * batch_size
            sigma_running += float(sigma.mean()) * batch_size
            samples += batch_size

        train_loss = running / max(samples, 1)
        val_source = val_loader if val_loader is not None else train_loader
        val_clean, val_robust = evaluate(
            rae, val_source, device, use_amp, args.latitude_weighted_loss, args.noise_tau
        )
        row = {
            "epoch": epoch,
            "train_robust_loss": train_loss,
            "val_clean_loss": val_clean,
            "val_robust_loss": val_robust,
            "mean_sigma": sigma_running / max(samples, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        print(row)
        append_jsonl(str(output / "log.jsonl"), row)

        payload = {
            "rae": rae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_robust_val": min(best_robust, val_robust),
            "args": vars(args),
            "noise_robust": True,
        }
        save_checkpoint(str(output / "checkpoint_latest.pt"), payload)
        if val_robust < best_robust:
            best_robust = val_robust
            payload["best_robust_val"] = best_robust
            save_checkpoint(str(output / "checkpoint_best.pt"), payload)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(str(output / f"checkpoint_epoch_{epoch + 1:04d}.pt"), payload)

        if (epoch + 1) % args.preview_every == 0:
            preview = next(iter(val_source))
            views, angles, gauge, _ = unpack_batch(preview)
            views = views[:1].to(device)
            angles = angles[:1].to(device)
            gauge = gauge[:1].to(device) if gauge is not None else None
            rae.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                z = rae.encode(views, angles, gauge)
                noisy_z, _ = add_stochastic_latent_noise(z, args.noise_tau)
                pred = rae.decode(noisy_z, angles, gauge)
            save_panorama_batch(
                pred.float(), angles, str(output / "previews"),
                f"epoch_{epoch + 1:04d}_robust", args.pano_h, args.pano_w,
                args.grid_height, target_views=views.float(), save_tangent_grid=False,
            )


if __name__ == "__main__":
    main(get_parser().parse_args())
