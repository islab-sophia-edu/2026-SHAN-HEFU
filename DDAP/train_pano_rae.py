"""Stage 1: adapt the PanoMAE decoder to full normalized encoder tokens."""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import (
    append_jsonl,
    build_rae_from_pretrain,
    save_checkpoint,
    save_panorama_batch,
    seed_everything,
    unpack_batch,
)
from pano_rae import latitude_weighted_mse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train PanoMAE representation decoder")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_data_path", default="")
    parser.add_argument("--pano_checkpoint", required=True)
    parser.add_argument("--output_dir", default="./output_pano_rae")

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

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--latent_noise_std", type=float, default=0.0,
                        help="Optional decoder robustness training; keep 0.0 for the clean baseline.")
    parser.add_argument("--latitude_weighted_loss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aug_device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview_every", type=int, default=10)
    parser.add_argument("--resume", default="")
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


@torch.no_grad()
def evaluate(rae, loader, device, use_amp: bool, weighted: bool) -> float:
    rae.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        views, angles, gauge, weights = unpack_batch(batch)
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        gauge = gauge.to(device, non_blocking=True) if gauge is not None else None
        weights = weights.to(device, non_blocking=True) if weighted else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            z = rae.encode(views, angles, gauge)
            pred = rae.decode(z, angles, gauge)
            loss = latitude_weighted_mse(pred, views, weights)
        batch_size = views.shape[0]
        total_loss += float(loss) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def main(args) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.aug_device == "cuda" and args.num_workers != 0:
        raise ValueError(
            "The uploaded dataset performs CUDA grid_sample inside __getitem__. "
            "Use --num_workers 0 with --aug_device cuda, or use --aug_device cpu."
        )

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(args.data_path, args)
    val_set = make_dataset(args.val_data_path, args) if args.val_data_path else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_set is not None else None
    )

    rae = build_rae_from_pretrain(args, device)
    rae.freeze_encoder()
    trainable = [p for p in rae.parameters() if p.requires_grad]
    print(f"RAE latent: N={2 * args.grid_height * args.grid_height}, D={rae.latent_dim}")
    print(f"Trainable decoder parameters: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_iter)
    total_steps = max(args.epochs * updates_per_epoch, 1)
    warmup_steps = args.warmup_epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(
            step,
            total_steps,
            warmup_steps,
            args.min_lr / args.lr,
        ),
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        rae.load_state_dict(checkpoint["rae"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))

    global_step = start_epoch * updates_per_epoch
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        rae.train(True)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        samples = 0

        for step, batch in enumerate(train_loader):
            views, angles, gauge, weights = unpack_batch(batch)
            views = views.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            gauge = gauge.to(device, non_blocking=True) if gauge is not None else None
            weights = weights.to(device, non_blocking=True) if args.latitude_weighted_loss else None

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    z = rae.encode(views, angles, gauge)
            if args.latent_noise_std > 0:
                z = z + args.latent_noise_std * torch.randn_like(z)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = rae.decode(z, angles, gauge)
                loss = latitude_weighted_mse(pred, views, weights)
                scaled_loss = loss / args.accum_iter

            scaler.scale(scaled_loss).backward()
            do_update = ((step + 1) % args.accum_iter == 0) or (step + 1 == len(train_loader))
            if do_update:
                scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += float(loss.detach()) * views.shape[0]
            samples += views.shape[0]

        train_loss = running / max(samples, 1)
        val_loss = evaluate(rae, val_loader, device, use_amp, args.latitude_weighted_loss) if val_loader else train_loss
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        print(row)
        append_jsonl(str(output / "log.jsonl"), row)

        payload = {
            "rae": rae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
            "args": vars(args),
        }
        save_checkpoint(str(output / "checkpoint_latest.pt"), payload)
        if val_loss < best_val:
            best_val = val_loss
            payload["best_val"] = best_val
            save_checkpoint(str(output / "checkpoint_best.pt"), payload)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(str(output / f"checkpoint_epoch_{epoch + 1:04d}.pt"), payload)

        if (epoch + 1) % args.preview_every == 0:
            preview_batch = next(iter(val_loader if val_loader else train_loader))
            views, angles, gauge, _ = unpack_batch(preview_batch)
            views = views[:2].to(device)
            angles = angles[:2].to(device)
            gauge = gauge[:2].to(device) if gauge is not None else None
            rae.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                pred = rae.decode(rae.encode(views, angles, gauge), angles, gauge)
            save_panorama_batch(
                pred.float(),
                angles,
                str(output / "previews"),
                f"epoch_{epoch + 1:04d}",
                args.pano_h,
                args.pano_w,
                args.grid_height,
            )


if __name__ == "__main__":
    main(get_parser().parse_args())
