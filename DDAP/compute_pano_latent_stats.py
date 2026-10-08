"""Compute fixed-position/channel statistics for Stage-2 PanoMAE latents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import build_rae_from_pretrain, load_rae_checkpoint, seed_everything, unpack_batch
from latent_stats import OnlinePositionChannelStats


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Compute PanoMAE latent statistics")
    p.add_argument("--data_path", required=True)
    p.add_argument("--pano_checkpoint", required=True)
    p.add_argument("--rae_checkpoint", required=True)
    p.add_argument("--output", required=True, help="Output .pt latent statistics")
    p.add_argument("--pano_model", default="vit_huge_patch14",
                   choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"])
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
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 means the entire training set")
    p.add_argument("--device", default="cuda")
    p.add_argument("--aug_device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-6)
    return p


def make_dataset(args):
    return PanoramicDataset(
        root_dir=args.data_path,
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


def main(args) -> None:
    if args.aug_device == "cuda" and args.num_workers != 0:
        raise ValueError("Use --num_workers 0 with CUDA tangent extraction")
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = make_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )

    rae = build_rae_from_pretrain(args, device)
    load_rae_checkpoint(rae, args.rae_checkpoint, strict=True)
    rae.freeze_all()

    token_count = 2 * args.grid_height * args.grid_height
    moments = OnlinePositionChannelStats(token_count, rae.latent_dim)
    processed = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            views, angles, gauge, _ = unpack_batch(batch)
            views = views.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            gauge = gauge.to(device, non_blocking=True) if gauge is not None else None
            z = rae.encode(views, angles, gauge).float()

            if args.max_samples > 0 and processed + z.shape[0] > args.max_samples:
                z = z[: args.max_samples - processed]
            moments.update(z)
            processed += int(z.shape[0])

            if batch_index % 50 == 0:
                print(f"latent stats: processed {processed}/{len(dataset)}")
            if args.max_samples > 0 and processed >= args.max_samples:
                break

    stats = moments.finalize(eps=args.eps)
    stats.save(args.output)
    standardized_mean = float(stats.normalize(stats.mean_cpu).mean())
    summary = {
        "path": str(Path(args.output).resolve()),
        "count": stats.count,
        "shape": [1, stats.token_count, stats.latent_dim],
        "fingerprint": stats.fingerprint,
        "mean_abs_average": float(stats.mean_cpu.abs().mean()),
        "std_average": float(stats.std_cpu.mean()),
        "std_min": float(stats.std_cpu.min()),
        "std_max": float(stats.std_cpu.max()),
        "sanity_standardized_mean_of_mean_tensor": standardized_mean,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(get_parser().parse_args())
