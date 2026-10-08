"""Stage-2 diagnostics: real-latent denoising, statistics and ERP-only output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from util.datasets_PanoMAE_gctt import PanoramicDataset
from common import (
    build_rae_from_pretrain,
    canonical_geometry,
    load_rae_checkpoint,
    save_panorama_batch,
    seed_everything,
    unpack_batch,
)
from flow_transport import DimensionShiftedRectifiedFlow
from latent_stats import LatentStandardizer
from pano_dit import PanoDiTDDT
from sample_pano_dit import restore_training_args
from train_pano_dit import OBJECTIVE


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Diagnose PanoDiT v4")
    p.add_argument("--data_path", required=True)
    p.add_argument("--dit_checkpoint", required=True)
    p.add_argument("--pano_checkpoint", required=True)
    p.add_argument("--rae_checkpoint", required=True)
    p.add_argument("--output_dir", default="./diagnose_pano_dit_v4")
    p.add_argument("--weights", choices=["raw", "ema"], default="raw")
    p.add_argument("--base_times", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    p.add_argument("--device", default="cuda")
    p.add_argument("--aug_device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_non_strict_pano_load", action="store_true")
    return p


def dataset_for(args):
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


def tensor_stats(x: torch.Tensor) -> dict:
    xf = x.float()
    return {
        "mean": float(xf.mean()),
        "std": float(xf.std(unbiased=False)),
        "abs_max": float(xf.abs().max()),
        "finite": bool(torch.isfinite(xf).all()),
    }


def main(cli) -> None:
    if cli.aug_device == "cuda" and cli.num_workers != 0:
        raise ValueError("Use --num_workers 0 with CUDA tangent extraction")
    seed_everything(cli.seed)
    device = torch.device(cli.device)
    ckpt = torch.load(cli.dit_checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("objective") != OBJECTIVE:
        raise RuntimeError("Diagnostic requires a v4 standardized-latent checkpoint")
    args = restore_training_args(cli, ckpt["args"])
    args.device = str(device)
    args.aug_device = cli.aug_device
    args.num_workers = cli.num_workers

    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    standardizer = LatentStandardizer.from_payload(ckpt["latent_stats"])
    rae = build_rae_from_pretrain(args, device)
    load_rae_checkpoint(rae, cli.rae_checkpoint, strict=True)
    rae.freeze_all()
    token_count = 2 * args.grid_height * args.grid_height
    standardizer.validate_shape(token_count, rae.latent_dim)

    base_width = rae.latent_dim if args.base_hidden_size == 0 else args.base_hidden_size
    model = PanoDiTDDT(
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
    model.load_state_dict(ckpt["ema" if cli.weights == "ema" else "dit"], strict=True)
    model.eval()
    transport = DimensionShiftedRectifiedFlow(
        token_count, rae.latent_dim,
        base_dimension=args.base_flow_dimension,
        time_eps=args.time_eps,
    )

    loader = DataLoader(dataset_for(args), batch_size=1, shuffle=False,
                        num_workers=cli.num_workers, pin_memory=device.type == "cuda")
    views, angles, gauge, _ = unpack_batch(next(iter(loader)))
    views = views.to(device)
    angles = angles.to(device)
    gauge = gauge.to(device) if gauge is not None else None

    report = {
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)),
        "weights": cli.weights,
        "stats_fingerprint": standardizer.fingerprint,
        "tests": {},
    }

    generator = torch.Generator(device=device).manual_seed(cli.seed + 98765)
    with torch.inference_mode():
        z_decoder = rae.encode(views, angles, gauge).float()
        x0 = standardizer.normalize(z_decoder)
        clean_pred = rae.decode(z_decoder, angles, gauge)
        save_panorama_batch(
            clean_pred.float(), angles, str(out), "clean_decoder",
            args.pano_h, args.pano_w, args.grid_height,
            target_views=views.float(), save_tangent_grid=False,
        )
        report["z_decoder"] = tensor_stats(z_decoder)
        report["x0_standardized"] = tensor_stats(x0)

        noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
        for base_t in cli.base_times:
            if not 0.0 < base_t < 1.0:
                raise ValueError("base_times must lie in (0,1)")
            actual_t = transport.shift_time(
                torch.tensor([base_t], device=device, dtype=x0.dtype)
            )
            t_view = actual_t.view(1, 1, 1)
            x_t = (1.0 - t_view) * x0 + t_view * noise
            target_v = noise - x0
            pred_v = model(x_t, actual_t, angles, gauge)
            velocity_mse = float((pred_v.float() - target_v.float()).square().mean())
            zero_mse = float(target_v.float().square().mean())

            # For the linear path x_t=x0+t*v, estimate x0 directly from one model call.
            x0_hat = x_t - t_view * pred_v
            z_hat = standardizer.denormalize(x0_hat.float())
            pred_views = rae.decode(z_hat, angles, gauge)
            tag = f"real_noisy_t{base_t:.2f}".replace(".", "p")
            save_panorama_batch(
                pred_views.float(), angles, str(out), tag,
                args.pano_h, args.pano_w, args.grid_height,
                target_views=views.float(), save_tangent_grid=False,
            )
            report["tests"][tag] = {
                "base_time": base_t,
                "shifted_time": float(actual_t),
                "velocity_mse": velocity_mse,
                "zero_predictor_mse": zero_mse,
                "relative_to_zero": velocity_mse / max(zero_mse, 1e-12),
                "pred_velocity": tensor_stats(pred_v),
                "x0_hat": tensor_stats(x0_hat),
            }

        # Pure-noise generation is only meaningful after the real-noisy tests pass.
        sample_angles, sample_gauge, _ = canonical_geometry(args.grid_height, 1, device)
        initial_noise = torch.randn((1, token_count, rae.latent_dim),
                                    generator=generator, device=device)
        x_sample = transport.sample(
            model, initial_noise.shape, sample_angles,
            sample_gauge if args.use_gctt else None,
            num_steps=cli.sampling_steps, solver=cli.solver,
            initial_noise=initial_noise,
        )
        z_sample = standardizer.denormalize(x_sample.float())
        sample_views = rae.decode(z_sample, sample_angles,
                                  sample_gauge if args.use_gctt else None)
        save_panorama_batch(
            sample_views.float(), sample_angles, str(out), "pure_noise_sample",
            args.pano_h, args.pano_w, args.grid_height,
            target_views=None, save_tangent_grid=False,
        )
        report["pure_noise_x"] = tensor_stats(x_sample)

    with open(out / "diagnostic_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"ERP-only diagnostics written to {out.resolve()}")


if __name__ == "__main__":
    main(parser().parse_args())
