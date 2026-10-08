"""Evaluate the clean Stage-1 RAE reconstruction ceiling on a validation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import build_rae_from_pretrain, load_rae_checkpoint, seed_everything, unpack_batch
from pano_metrics import compute_paired_metrics


METRIC_KEYS = (
    "tangent_psnr",
    "tangent_ssim",
    "erp_psnr",
    "erp_ssim",
    "erp_ws_psnr",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate PanoRAE reconstruction PSNR/SSIM")
    p.add_argument("--data_path", required=True)
    p.add_argument("--pano_checkpoint", required=True)
    p.add_argument("--rae_checkpoint", required=True)
    p.add_argument("--output", default="./pano_rae_metrics.json")
    p.add_argument("--pano_model", default="vit_huge_patch14")
    p.add_argument("--pano_h", type=int, default=1024)
    p.add_argument("--pano_w", type=int, default=2048)
    p.add_argument("--grid_height", type=int, default=16)
    p.add_argument("--geometric_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_angular_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--angular_bias_init_slope", type=float, default=1.0)
    p.add_argument("--use_gctt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_gauge_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gauge_bias_init", type=float, default=0.0)
    p.add_argument("--allow_non_strict_pano_load", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 evaluates the full dataset")
    p.add_argument("--metric_chunk_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aug_device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
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
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    rae = build_rae_from_pretrain(args, device)
    ckpt = load_rae_checkpoint(rae, args.rae_checkpoint, strict=True)
    rae.freeze_all()

    totals = {key: 0.0 for key in METRIC_KEYS}
    processed = 0

    with torch.inference_mode():
        for batch in loader:
            views, angles, gauge, _ = unpack_batch(batch)
            views = views.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            gauge = gauge.to(device, non_blocking=True) if gauge is not None else None

            z = rae.encode(views, angles, gauge).float()
            pred = rae.decode(z, angles, gauge)
            metrics = compute_paired_metrics(
                pred.float(),
                views.float(),
                angles,
                args.pano_h,
                args.pano_w,
                args.grid_height,
                tangent_chunk_size=args.metric_chunk_size,
            )
            batch_n = int(views.shape[0])
            for key in METRIC_KEYS:
                totals[key] += float(metrics[key]) * batch_n
            processed += batch_n
            print({"processed": processed, **metrics})

            if args.max_samples > 0 and processed >= args.max_samples:
                break

    result = {
        "count": processed,
        "noise_robust": bool(ckpt.get("noise_robust", False)),
        "rae_checkpoint": args.rae_checkpoint,
        "metrics": {key: totals[key] / max(processed, 1) for key in METRIC_KEYS},
        "note": (
            "ERP target is reconstructed from the same GT tangent views using the "
            "same inverse-ODI operation; this isolates model reconstruction from "
            "projection asymmetry."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(parser().parse_args())
