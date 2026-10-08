"""Sample inverse-ODI ERP panoramas from standardized-latent PanoDiT."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from common import (
    build_rae_from_pretrain,
    canonical_geometry,
    load_rae_checkpoint,
    save_panorama_batch,
    seed_everything,
)
from flow_transport import DimensionShiftedRectifiedFlow
from latent_stats import LatentStandardizer
from pano_dit import PanoDiTDDT
from train_pano_dit import OBJECTIVE


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Sample stable PanoDiT v7")
    p.add_argument("--dit_checkpoint", required=True)
    p.add_argument("--pano_checkpoint", required=True)
    p.add_argument("--rae_checkpoint", required=True)
    p.add_argument("--output_dir", default="./pano_dit_v7_samples")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_non_strict_pano_load", action="store_true")
    return p


def restore_training_args(cli, saved: dict):
    fields = {
        "pano_model", "pano_h", "pano_w", "grid_height",
        "geometric_bias", "use_angular_bias", "angular_bias_init_slope",
        "use_gctt", "use_gauge_bias", "gauge_bias_init",
        "base_hidden_size", "base_depth", "base_num_heads",
        "head_hidden_size", "head_depth", "head_num_heads",
        "head_use_geometry", "mlp_ratio", "dropout",
        "base_flow_dimension", "time_eps",
    }
    missing = sorted(fields - saved.keys())
    if missing:
        raise KeyError(f"Checkpoint is missing training fields: {missing}")
    for field in fields:
        setattr(cli, field, saved[field])
    return cli


def main(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.dit_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("objective") != OBJECTIVE:
        raise RuntimeError(
            "Checkpoint is not the stable v7 standardized-latent flow model."
        )
    args = restore_training_args(args, checkpoint["args"])
    args.device = str(device)

    standardizer = LatentStandardizer.from_payload(checkpoint["latent_stats"])
    if standardizer.fingerprint != checkpoint.get("latent_stats_fingerprint"):
        raise RuntimeError("Checkpoint latent stats fingerprint mismatch")

    rae = build_rae_from_pretrain(args, device)
    load_rae_checkpoint(rae, args.rae_checkpoint, strict=True)
    rae.freeze_all()
    token_count = 2 * args.grid_height * args.grid_height
    standardizer.validate_shape(token_count, rae.latent_dim)

    base_width = rae.latent_dim if args.base_hidden_size == 0 else args.base_hidden_size
    dit = PanoDiTDDT(
        latent_dim=rae.latent_dim,
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
    ).to(device)
    key = "ema" if args.use_ema and "ema" in checkpoint else "dit"
    dit.load_state_dict(checkpoint[key], strict=True)
    dit.eval()

    transport = DimensionShiftedRectifiedFlow(
        token_count, rae.latent_dim,
        base_dimension=args.base_flow_dimension,
        time_eps=args.time_eps,
        time_distribution=getattr(args, "time_distribution", "logit_normal"),
        logit_normal_mean=getattr(args, "logit_normal_mean", 0.0),
        logit_normal_std=getattr(args, "logit_normal_std", 1.0),
    )
    angles, gauge, _ = canonical_geometry(args.grid_height, args.num_samples, device)
    shape = (args.num_samples, token_count, rae.latent_dim)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(shape, generator=generator, device=device)

    with torch.inference_mode():
        z_standardized = transport.sample(
            dit, shape, angles, gauge if args.use_gctt else None,
            num_steps=args.sampling_steps,
            solver=args.solver,
            initial_noise=initial_noise,
        )
        if not torch.isfinite(z_standardized).all():
            raise FloatingPointError("Generated latent contains NaN/Inf")
        z_decoder = standardizer.denormalize(z_standardized.float())
        predicted_views = rae.decode(z_decoder, angles, gauge if args.use_gctt else None)

    save_panorama_batch(
        predicted_views.float(), angles, args.output_dir, "sample",
        args.pano_h, args.pano_w, args.grid_height,
        target_views=None, save_tangent_grid=False,
    )
    print(f"Loaded common.py: {inspect.getfile(save_panorama_batch)}")
    print(f"Loaded stats fingerprint: {standardizer.fingerprint}")
    print(f"Saved ERP-only files to: {Path(args.output_dir).resolve()}")
    print("Expected filename pattern: sample_XXX_pred_erp.png")


if __name__ == "__main__":
    main(get_parser().parse_args())
