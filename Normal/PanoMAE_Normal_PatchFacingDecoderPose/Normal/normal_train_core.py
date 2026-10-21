"""GCTT normal fine-tuning (official Stanford pano/normal GT)."""
import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import normal_model_core as models_normal
import normal_engine_core as engine_normal
from normal_dataset_core import build_normal_dataset
import normal_lrd as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("PanoMAE GCTT Normal Estimation Official GT", add_help=False)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=4, type=int)
    parser.add_argument("--model", default="vit_base_patch16", type=str, metavar="MODEL")
    parser.add_argument("--pano_h", default=2048, type=int)
    parser.add_argument("--pano_w", default=4096, type=int)
    parser.add_argument("--grid_height", default=16, type=int)
    parser.add_argument("--in_chans", default=3, type=int, help="3=RGB only, 6=RGB + per-pixel rayXYZ. Default 3 in v8; rayXYZ is intentionally disabled for the minimal encoder-adapter run.")
    parser.add_argument("--ray_coord", default="world", choices=["world", "local"], help="Coordinate frame for rayXYZ channels. world matches GCTT/ERP frame and makes ray token informative; local matches patch camera frame.")
    parser.add_argument("--no_ray_token", action="store_false", dest="use_ray_token", help="Disable token-level ray embedding while keeping rayXYZ input channels.")
    parser.set_defaults(use_ray_token=False)
    parser.add_argument("--ray_embed_scale_init", default=0.1, type=float)

    # Normal convention / handedness.
    parser.add_argument("--normal_folder_name", default="normal", type=str)
    parser.add_argument("--depth_folder_name", default="depth", type=str)
    parser.add_argument("--normal_label_source", default="normal", choices=["normal"],
                        help="depth: derive local/global normal target from Stanford depth with the exact GCTT tangent projection; normal: use official Stanford normal PNG. Default depth in v9 to remove normal-RGB convention ambiguity.")
    parser.add_argument("--depth_scale", default=512.0, type=float)
    parser.add_argument("--invalid_depth_raw", default=65535.0, type=float)
    parser.add_argument("--depth_valid_min", default=0.05, type=float)
    parser.add_argument("--depth_valid_max", default=0.0, type=float, help="0 disables upper filtering; valid far depth is kept and only missing raw is rejected.")
    parser.add_argument("--depth_normal_smooth", default=0.0, type=float)
    parser.add_argument("--invalid_normal_value", default=128, type=int,
                        help="Stanford missing normal color is #808080.")
    parser.add_argument("--invalid_normal_tolerance", default=1.0, type=float)
    parser.add_argument("--normal_axis_perm", default="0,2,1", type=str,
                        help="Permutation applied to decoded Stanford RGB normal channels. Default uses the Stanford/NYU channel order directly. Use 0,2,1 to reproduce the old code.")
    parser.add_argument("--normal_axis_sign", default="-1,-1,1", type=str,
                        help="Signs after axis permutation. Use for handedness ablation, e.g. 1,-1,1.")
    parser.add_argument("--normal_local_y_sign", default=1.0, type=float,
                        help="+1 means target local normal uses the same GCTT frame as SPE: [x_g, y_g, n_c]. Use -1 only to reproduce the old image-down camera convention [x_g, -y_g, n_c].")
    parser.add_argument("--official_normal_coordinate", default="erp", choices=["erp"],
                        help="Convert official Stanford world Normal GT into camera-aligned ERP frame using Pose JSON.")
    parser.add_argument("--normal_target_frame", default="global", choices=["global"],
                        help="Default global: train on Stanford pano global normals. local: project targets into GCTT [x_g,y_g,n_c].")
    parser.add_argument("--w_cos", default=1.0, type=float)
    parser.add_argument("--w_l1", default=0.0, type=float)
    parser.add_argument("--normal_prediction_frame", default="stanford",
                        choices=["stanford", "legacy_yaw180"],
                        help="Signed prediction coordinate frame. stanford: direct output (default). "
                             "legacy_yaw180: rotate raw predicted X/Z signs before training loss "
                             "and evaluation; only for a verified legacy checkpoint convention.")
    parser.add_argument("--allow_frame_override", action="store_true",
                        help="Explicitly override recorded checkpoint prediction-frame mode; diagnostic use only.")
    parser.add_argument("--normal_loss_bidirectional", action="store_true", help="Use min(angle(n,t), angle(n,-t)) loss only as a diagnostic/robust option.")
    parser.add_argument("--normal_metric_bidirectional", action="store_true", help="Report sign-invariant metrics; oriented metrics are still reported by default.")
    parser.add_argument("--no_normal_convention_sweep", action="store_false", dest="normal_convention_sweep", help="Disable validation-time axis/sign convention sweep.")
    parser.set_defaults(normal_convention_sweep=False)
    parser.add_argument("--normal_convention_sweep_batches", default=3, type=int)
    parser.add_argument("--normal_convention_sweep_max_pixels", default=200000, type=int)

    # Normal-specific encoder adaptation, memory-light.
    parser.add_argument("--depth_encoder_depth", default=0, type=int,
                        help="Default 0 in v8 to keep your original compact decoder path. Set >0 only for ablation.")
    parser.add_argument("--stage_encoder_depth", default=0, type=int)
    parser.add_argument("--depth_encoder_mlp_ratio", default=0.5, type=float)
    parser.add_argument("--depth_encoder_kernel_size", default=3, type=int)
    parser.add_argument("--depth_encoder_drop_path", default=0.0, type=float)
    parser.add_argument("--depth_encoder_init_scale", default=1e-2, type=float)
    parser.add_argument("--cnn_base_dim", default=64, type=int)
    parser.add_argument("--adapter_lr_mult", default=1.0, type=float)
    parser.add_argument("--encoder_lr_mult", default=1.0, type=float,
                        help="Optional multiplier for pretrained GCTT encoder params. Use >1 only to test whether pretrained encoder is under-updated.")
    parser.add_argument("--no_adapter_lr_mult", action="store_true")
    parser.add_argument("--no_encoder_adapters", action="store_false", dest="use_encoder_adapters",
                        help="Disable low-rank residual adapters inside every GCTT encoder block.")
    parser.set_defaults(use_encoder_adapters=False)
    parser.add_argument("--encoder_adapter_dim", default=64, type=int)
    parser.add_argument("--encoder_adapter_init_scale", default=1e-3, type=float)
    parser.add_argument("--no_aux_normal", action="store_false", dest="use_aux_normal",
                        help="Disable direct encoder auxiliary normal head.")
    parser.set_defaults(use_aux_normal=False)
    parser.add_argument("--w_aux_normal", default=0.3, type=float,
                        help="Deep supervision weight for the encoder-attached auxiliary normal head.")

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--clip_grad", type=float, default=1.0, metavar="NORM")
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")
    parser.add_argument("--dist_eval", action="store_true", default=False)

    parser.add_argument("--data_path", default="./train", type=str)
    parser.add_argument("--val_data_path", default="./test", type=str)
    parser.add_argument("--train_areas", default="area_1,area_2,area_3,area_4,area_6", type=str)
    parser.add_argument("--val_areas", default="area_5a", type=str)
    parser.add_argument("--rgb_folder_name", default="rgb", type=str)
    parser.add_argument("--debug_limit", default=None, type=int)
    parser.add_argument("--diagnostic_train_eval", action="store_true",
                        help="Evaluate the first 2 training-area panoramas in deterministic eval mode; diagnostics only.")

    # GCTT / oriented SPE options.
    parser.add_argument("--no_gctt", action="store_false", dest="use_gctt")
    parser.set_defaults(use_gctt=True)
    parser.add_argument("--gctt_gauge_jitter_deg", default=30.0, type=float)
    parser.add_argument("--gctt_local_gauge_jitter_deg", default=0.0, type=float)
    parser.add_argument("--no_gauge_bias", action="store_false", dest="use_gauge_bias")
    parser.set_defaults(use_gauge_bias=True)
    parser.add_argument("--gauge_bias_init", default=0.0, type=float)
    parser.add_argument("--no_angular_bias", action="store_false", dest="use_angular_bias")
    parser.set_defaults(use_angular_bias=True)
    parser.add_argument("--angular_bias_init_slope", default=1.0, type=float)
    parser.add_argument("--no_geometric_bias", action="store_false", dest="geometric_bias")
    parser.set_defaults(geometric_bias=True)

    # Train augmentations. Dataset automatically disables these for val/test.
    parser.add_argument("--angle_jitter_deg", default=0.0, type=float)
    parser.add_argument("--no_color_jitter", action="store_false", dest="use_color_jitter")
    parser.set_defaults(use_color_jitter=True)
    parser.add_argument("--no_blur", action="store_false", dest="use_blur")
    parser.set_defaults(use_blur=True)
    parser.add_argument("--use_full_pose3d", action="store_true", dest="use_full_pose3d",
                        help="Enable yaw/pitch/roll ERP augmentation for normal. Disabled by default until normal coordinate convention is validated.")
    parser.set_defaults(use_full_pose3d=False)
    parser.add_argument("--pose_yaw_deg", default=360.0, type=float)
    parser.add_argument("--pose_pitch_deg", default=30.0, type=float)
    parser.add_argument("--pose_roll_deg", default=30.0, type=float)
    parser.add_argument("--no_horizontal_roll", action="store_false", dest="use_horizontal_roll")
    parser.add_argument("--use_horizontal_roll", action="store_true", dest="use_horizontal_roll")
    parser.set_defaults(use_horizontal_roll=False)
    parser.add_argument("--hflip_prob", default=0.0, type=float)
    parser.add_argument("--aug_device", default="cpu", type=str)

    # Decoder-only camera geometry. Baseline model remains unchanged.
    parser.add_argument("--no_decoder_camera_pose", action="store_false", dest="decoder_camera_pose",
                        help="Remove decoder camera pose for a controlled experiment; encoder is unchanged.")
    parser.set_defaults(decoder_camera_pose=True)
    parser.add_argument("--decoder_pose_strength", default=0.2, type=float,
                        help="FiLM/ray conditioning scale. Default 0.2; disable pose using --no_decoder_camera_pose.")
    parser.add_argument("--no_decoder_pose_output_rotation", action="store_false",
                        dest="decoder_pose_rotate_output",
                        help="FiLM-only diagnostic: do not rotate tangent camera-local Normal output to ERP.")
    parser.set_defaults(decoder_pose_rotate_output=True)

    # Visualization.
    parser.add_argument("--save_normal_vis_every", default=0, type=int)
    parser.add_argument("--inverse_odi_blend", default="average", choices=["overwrite", "average", "maxweight"],
                        help="Normal inverse-ODI stitching for visualization. average uses center-weighted vector blending and is the default because it suppresses patch-boundary seams; maxweight is only for diagnostics.")
    parser.add_argument("--inverse_odi_fov_scale", default=1.02, type=float,
                        help="Visualization-only inverse-ODI FOV expansion. Values slightly above 1 fill cell-boundary gaps caused by exact tangent-grid round trips. Does not affect training or metrics.")
    parser.add_argument("--inverse_odi_valid_dilate", default=5, type=int,
                        help="Visualization-only dilation for strict normal valid masks. Fills thin border gaps without changing loss/metrics.")
    parser.add_argument("--vis_normal_error_max", default=90.0, type=float,
                        help="Kept for backward compatibility; the default saved comparison now contains only pred and gt.")
    parser.add_argument("--save_patch_grid_vis", action="store_true")

    parser.add_argument("--output_dir", default="./output_normal")
    parser.add_argument("--log_dir", default="./output_normal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--finetune", default="", help="Path to a matching variant-specific MAE pretrained checkpoint")
    parser.add_argument("--allow_random_init", action="store_true", help="Diagnostic only: run without a pretrained checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def _adapt_view_embed_weight(value, target_shape):
    target_out, target_in, target_h, target_w = target_shape
    src_out, src_in, src_h, src_w = value.shape
    if src_out != target_out:
        return None
    if src_in != target_in:
        if target_in > src_in:
            pad = torch.zeros(src_out, target_in - src_in, src_h, src_w, dtype=value.dtype)
            value = torch.cat([value, pad], dim=1)
        else:
            value = value[:, :target_in]
    if value.shape[2:] != (target_h, target_w):
        value = F.interpolate(value, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return value


def load_pretrained_weights(model, checkpoint_path):
    if misc.is_main_process():
        print("============== Loading GCTT Pretrained Weights ==============")
        print(f"Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    def get_saved(key):
        return saved_args.get(key) if isinstance(saved_args, dict) else getattr(saved_args, key, None)
    variant = getattr(model, "normal_variant", None)
    adaptive = get_saved("adaptive_masking")
    dynamic = get_saved("dynamic_mask_ratio")
    ratio = get_saved("mask_ratio")
    if variant == "noadaptive" and (adaptive is True or dynamic is True or
                                    (ratio is not None and abs(float(ratio) - 0.75) > 1e-6)):
        raise RuntimeError("Fixed-0.75 ablation checkpoint is NOT fixed/unadaptive according to its pretrain metadata")
    if variant in {"gctt", "grid"} and adaptive is False:
        raise RuntimeError("Adaptive PanoMAE checkpoint metadata reports adaptive_masking=False")
    if variant in {"noadaptive", "gctt", "grid"} and adaptive is None:
        print("[CHECKPOINT WARNING] Adaptive masking metadata missing; verify pretrain logs before using results")
    model_dict = model.state_dict()
    source_fourier = next((v for k, v in state_dict.items()
                           if k.endswith("angle_pos_embed.fourier_weights")), None)
    target_fourier = model_dict.get("angle_pos_embed.fourier_weights", None)
    if (source_fourier is None) != (target_fourier is None):
        raise RuntimeError("Pretraining checkpoint positional family mismatch: planar 2DPE versus SPE")
    if source_fourier is not None and tuple(source_fourier.shape) != tuple(target_fourier.shape):
        raise RuntimeError(f"Pretraining checkpoint positional frame mismatch: {tuple(source_fourier.shape)} != {tuple(target_fourier.shape)}")
    new_dict, loaded_keys = {}, []
    skip_prefixes = (
        "mask_token", "decoder_pred", "head", "depth_head", "normal_head",
        "global_detail", "cnn_proj", "fusion", "neck_conv", "decoder_neck", "skip", "up",
        "depth_encoder", "stage_adapters",
    )
    for key, value in state_dict.items():
        k = key[7:] if key.startswith("module.") else key
        if any(k == p or k.startswith(p) for p in skip_prefixes):
            continue
        if "patch_embed.proj" in k:
            k = k.replace("patch_embed.proj", "view_embed.proj.proj")
        elif k.startswith("patch_embed"):
            k = k.replace("patch_embed", "view_embed")
        if k not in model_dict:
            continue
        target_shape = model_dict[k].shape
        if hasattr(value, "shape") and value.shape != target_shape:
            if k == "view_embed.proj.proj.weight" and value.dim() == 4:
                adapted = _adapt_view_embed_weight(value, target_shape)
                if adapted is None:
                    continue
                if misc.is_main_process():
                    print(f"  [Adapt view_embed] {k}: {tuple(value.shape)} -> {tuple(target_shape)}")
                value = adapted
            else:
                if misc.is_main_process():
                    print(f"  [Skip mismatch] {k}: ckpt {tuple(value.shape)} != model {tuple(target_shape)}")
                continue
        new_dict[k] = value
        loaded_keys.append(k)
    required = [k for k in model_dict if k.startswith(("view_embed.", "blocks.", "norm.", "angle_pos_embed.")) or k == "cls_token"]
    covered = len(set(required) & set(new_dict)) / max(len(required), 1)
    if covered < .90:
        raise RuntimeError(f"Wrong pretrain variant/checkpoint? Encoder key coverage {covered:.1%} < 90%")
    print(f"[Pretrain] Encoder key coverage: {covered:.1%}")
    encoder_keys = {k for k in model_dict if k.startswith("blocks.")}
    encoder_loaded = encoder_keys.intersection(loaded_keys)
    encoder_coverage = len(encoder_loaded) / max(len(encoder_keys), 1)
    if encoder_coverage < 0.80:
        raise RuntimeError(
            f"Only {encoder_coverage:.1%} of ViT encoder block tensors matched the MAE checkpoint. "
            "Check --model architecture, pretraining variant, and checkpoint path."
        )
    if misc.is_main_process():
        print(f"[Pretraining encoder coverage] {len(encoder_loaded)}/{len(encoder_keys)} "
              f"({encoder_coverage:.1%})")
    msg = model.load_state_dict(new_dict, strict=False)
    if misc.is_main_process():
        print(f"[Success] Loaded {len(loaded_keys)} keys.")
        print(msg)


def _boost_task_param_groups(param_groups, model, adapter_lr_mult, encoder_lr_mult=1.0):
    task_prefixes = (
        "normal_head", "global_detail", "cnn_proj", "fusion", "neck_conv", "skip", "up", "decoder_camera_pose"
    )
    encoder_prefixes = ("view_embed", "cls_token", "angle_pos_embed", "enc_ang_bias", "blocks", "norm")
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    new_groups = []
    for group in param_groups:
        task_params, encoder_params, base_params = [], [], []
        for p in group.get("params", []):
            name = id_to_name.get(id(p), "")
            if name.startswith(task_prefixes):
                task_params.append(p)
            elif name.startswith(encoder_prefixes):
                encoder_params.append(p)
            else:
                base_params.append(p)
        if base_params:
            g = dict(group); g["params"] = base_params; new_groups.append(g)
        if encoder_params:
            g = dict(group); g["params"] = encoder_params; g["lr_scale"] = float(g.get("lr_scale", 1.0)) * float(encoder_lr_mult); new_groups.append(g)
        if task_params:
            g = dict(group); g["params"] = task_params; g["lr_scale"] = float(g.get("lr_scale", 1.0)) * float(adapter_lr_mult); new_groups.append(g)
    return new_groups


def _print_param_group_summary(optimizer, model):
    if not misc.is_main_process():
        return
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    print("[Optimizer groups]")
    for gi, group in enumerate(optimizer.param_groups[:20]):
        names = [id_to_name.get(id(p), "?") for p in group.get("params", [])[:3]]
        print(f"  group {gi:02d}: lr_scale={group.get('lr_scale', 1.0):.4g}, lr={group.get('lr', 0):.3e}, n={len(group.get('params', []))}, sample={names}")


VARIANTS = {
    "2dpe": {"gctt": False, "pb": False, "grid": False, "position": "2dpe"},
    "3dpe": {"gctt": False, "pb": False, "grid": False, "position": "spherical"},
    "pb": {"gctt": False, "pb": True, "grid": False, "position": "spherical"},
    "noadaptive": {"gctt": True, "pb": True, "grid": False, "position": "spherical"},
    "grid": {"gctt": True, "pb": True, "grid": True, "position": "spherical"},
    "gctt": {"gctt": True, "pb": True, "grid": False, "position": "spherical"},
    "mae": {"gctt": False, "pb": False, "grid": True, "position": "2dpe"},
}


def _lock_variant(args):
    variant = getattr(args, "variant", "gctt")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Normal variant: {variant}")
    conf = VARIANTS[variant]
    args.use_gctt = conf["gctt"]
    args.use_angular_bias = conf["pb"]
    args.use_gauge_bias = conf["gctt"]
    args.grid_tokenization = conf["grid"]
    args.position_mode = conf["position"]
    args.gctt_gauge_jitter_deg = 0.0
    args.gctt_local_gauge_jitter_deg = 0.0
    args.variant = variant
    if variant == "mae":
        args.decoder_camera_pose = False
    args.use_decoder_camera_pose = bool(args.decoder_camera_pose)
    if args.normal_loss_bidirectional or args.normal_metric_bidirectional:
        raise ValueError("Sign-invariant normal losses/metrics are excluded from the controlled study")
    if args.w_l1 or args.w_cos != 1.0:
        raise ValueError("Controlled study uses a single oriented cosine loss: w_cos=1, w_l1=0")
    if args.hflip_prob or args.use_full_pose3d or args.use_horizontal_roll:
        raise ValueError("Pose/flip/roll augmentation must remain disabled for this controlled comparison")
    return conf


def main(args):
    conf = _lock_variant(args)
    print(f"[Ablation] {args.variant}: T={not conf['grid']} SPE={conf['position']=='spherical'} PB={conf['pb']} GCTT={conf['gctt']}")
    if not args.eval and not args.resume and not args.finetune and not args.allow_random_init:
        raise ValueError("Controlled ablation requires --finetune with a matching pretrained checkpoint")
    if args.eval and not args.resume:
        raise ValueError("--eval requires a fully fine-tuned Normal checkpoint via --resume. --finetune only loads the pretraining encoder, leaving the Normal decoder randomly initialized.")
    if args.normal_prediction_frame == "legacy_yaw180" and misc.is_main_process():
        print("[FRAME WARNING] Explicit legacy_yaw180: rotating predicted X/Z signs "
              "for loss and metrics. Use ONLY if checkpoint frame provenance has been verified; "
              "never select this mode from held-out validation images.")
    if tuple(map(int, args.normal_axis_perm.split(","))) != (0, 2, 1) or \
       tuple(map(float, args.normal_axis_sign.split(","))) != (-1.0, -1.0, 1.0):
        print("[Normal Frame] Non-canonical custom perm/sign: verify against global_xyz before training.")
    else:
        print("[Normal Frame] Official RGB -> XYZ candidate: [-R, -B, +G] (requires global_xyz audit).")
    if args.normal_prediction_frame != "stanford":
        raise ValueError("Canonical XYZ candidate requires --normal_prediction_frame stanford; do not stack legacy_yaw180.")
    if args.normal_label_source != "normal" or args.normal_target_frame != "global" or args.official_normal_coordinate != "erp":
        raise ValueError("gctt: only Pose-aligned ERP normals from official pano/normal PNG are supported")
    print("[Normal Frame] Pose-aligned ERP supervision: Official PNG -> (-R,-B,+G) world -> inverse official Pose -> PanoMAE ERP")
    if args.num_workers > 0 and args.aug_device.startswith("cuda"):
        raise ValueError("CUDA in dataset workers is unsupported: set --aug_device cpu or --num_workers 0")
    if args.pano_h % args.grid_height or args.pano_w % (2 * args.grid_height):
        raise ValueError("ERP resolution must divide into a grid_height x 2*grid_height tile grid")
    train_areas = set(a.strip() for a in args.train_areas.split(',') if a.strip())
    val_areas = set(a.strip() for a in args.val_areas.split(',') if a.strip())
    if train_areas & val_areas:
        raise ValueError(f"Train/val leakage detected: {sorted(train_areas & val_areas)}")
    misc.init_distributed_mode(args)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU available")
        gpu = int(getattr(args, "gpu", 0)) if args.distributed else 0
        torch.cuda.set_device(gpu)
        args.device = f"cuda:{gpu}"
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); cudnn.benchmark = True

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    if misc.is_main_process():
        print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
        print(f"actual lr: {args.lr:.2e}")
        print(f"effective batch size: {eff_batch_size}")
        print(f"Normal axis perm/sign: {args.normal_axis_perm} / {args.normal_axis_sign}; local_y_sign={args.normal_local_y_sign}")
        print(f"Normal label source: {args.normal_label_source}; target frame: {args.normal_target_frame}")
        print("Encoder: ablation-specific pretraining backbone, RGB-only; shared dense decoder")
        print("Normal target: official pano/normal, world-to-ERP using official pose; depth unused")
        print(f"GCTT: {args.use_gctt}, global gauge jitter: {args.gctt_gauge_jitter_deg}, local gauge jitter: {args.gctt_local_gauge_jitter_deg}")
        print(f"Positional encoding: {args.position_mode}, GCTT={args.use_gctt}, PB={args.use_angular_bias}")
        print(f"Decoder camera pose: {args.use_decoder_camera_pose}, projection={'ERP' if conf['grid'] else 'tangent'}, strength={args.decoder_pose_strength}, rotate_output={args.decoder_pose_rotate_output and not conf['grid']}")
        print(f"Train augmentations: full_pose3d={args.use_full_pose3d}, blur={args.use_blur}, color_jitter={args.use_color_jitter}, horizontal_roll={args.use_horizontal_roll}; val disables them.")

    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    args.input_size = (patch_h, patch_w)

    dataset_train = build_normal_dataset(is_train=True, args=args)
    dataset_val = build_normal_dataset(is_train=False, args=args)
    diagnostic_loader = None
    if args.diagnostic_train_eval:
        diag_args = copy.copy(args)
        diag_args.val_areas = args.train_areas
        diag_args.debug_limit = 2
        diag_dataset = build_normal_dataset(is_train=False, args=diag_args)
        diag_sampler = (torch.utils.data.distributed.DistributedSampler(
            diag_dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(),
            shuffle=False, drop_last=False) if args.distributed else
            torch.utils.data.SequentialSampler(diag_dataset))
        diagnostic_loader = torch.utils.data.DataLoader(
            diag_dataset, sampler=diag_sampler, batch_size=1,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        print("[Diagnostic] Training-area eval uses 2 fixed samples, no augmentation. "
              "These results are not held-out validation metrics.")
    if misc.is_main_process():
        print(f"Train dataset size: {len(dataset_train)}")
        print(f"Val dataset size  : {len(dataset_val)}")
    if len(dataset_train) <= 0 or len(dataset_val) <= 0:
        raise ValueError("Dataset is empty. Check --data_path/--val_data_path and Stanford area folders.")

    if args.distributed:
        num_tasks = misc.get_world_size(); global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    if misc.is_main_process():
        print(f"Creating GCTT PanoNormal model: {args.model}")
    model = models_normal.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.input_size,
        in_chans=args.in_chans,
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
        geometric_bias=args.geometric_bias,
        position_mode=args.position_mode,
        use_gctt=args.use_gctt,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        use_gauge_bias=args.use_gauge_bias,
        gauge_bias_init=args.gauge_bias_init,
        depth_encoder_depth=args.depth_encoder_depth,
        stage_encoder_depth=args.stage_encoder_depth,
        depth_encoder_mlp_ratio=args.depth_encoder_mlp_ratio,
        depth_encoder_kernel_size=args.depth_encoder_kernel_size,
        depth_encoder_drop_path=args.depth_encoder_drop_path,
        depth_encoder_init_scale=args.depth_encoder_init_scale,
        cnn_base_dim=args.cnn_base_dim,
        use_ray_token=args.use_ray_token,
        ray_embed_scale_init=args.ray_embed_scale_init,
        use_encoder_adapters=args.use_encoder_adapters,
        encoder_adapter_dim=args.encoder_adapter_dim,
        encoder_adapter_init_scale=args.encoder_adapter_init_scale,
        use_aux_normal=args.use_aux_normal,
        use_decoder_camera_pose=args.use_decoder_camera_pose,
        decoder_pose_grid=conf["grid"],
        decoder_pose_strength=args.decoder_pose_strength,
        decoder_pose_rotate_output=args.decoder_pose_rotate_output,
    )
    model.normal_variant = args.variant
    if args.finetune and not args.resume:
        load_pretrained_weights(model, args.finetune)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
    if not args.no_adapter_lr_mult:
        param_groups = _boost_task_param_groups(param_groups, model_without_ddp, args.adapter_lr_mult, args.encoder_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    lrd.audit_stem_lr(model_without_ddp, optimizer, args.layer_decay,
                      args.encoder_lr_mult if not args.no_adapter_lr_mult else 1.0)
    _print_param_group_summary(optimizer, model_without_ddp)
    loss_scaler = NativeScaler()
    criterion = engine_normal.CosineNormalLoss(w_cos=args.w_cos, w_l1=args.w_l1, bidirectional=args.normal_loss_bidirectional).to(device)

    if args.resume:
        if misc.is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_args = checkpoint.get("args", None)
        saved_variant = (saved_args.get("variant") if isinstance(saved_args, dict)
                         else getattr(saved_args, "variant", None))
        saved_pose = (saved_args.get("use_decoder_camera_pose", False) if isinstance(saved_args, dict)
                      else getattr(saved_args, "use_decoder_camera_pose", False))
        if bool(saved_pose) != bool(args.use_decoder_camera_pose):
            raise ValueError(
                f"Decoder camera pose differs from checkpoint: saved={saved_pose}, "
                f"requested={args.use_decoder_camera_pose}. Use the matching model setting."
            )
        saved_rotate = (saved_args.get("decoder_pose_rotate_output", None) if isinstance(saved_args, dict)
                        else getattr(saved_args, "decoder_pose_rotate_output", None))
        if args.use_decoder_camera_pose and saved_rotate is not None and bool(saved_rotate) != bool(args.decoder_pose_rotate_output):
            raise ValueError("Fine-tuned checkpoint has a different decoder pose output-frame setting")
        saved_strength = (saved_args.get("decoder_pose_strength", None) if isinstance(saved_args, dict)
                          else getattr(saved_args, "decoder_pose_strength", None))
        if args.use_decoder_camera_pose and saved_strength is not None and abs(float(saved_strength) - args.decoder_pose_strength) > 1e-8:
            raise ValueError("Fine-tuned checkpoint has a different decoder pose conditioning strength")
        if saved_variant != args.variant:
            raise ValueError(f"Fine-tuned checkpoint variant mismatch: saved={saved_variant}, current={args.variant}")
        saved_coord = (saved_args.get("official_normal_coordinate") if isinstance(saved_args, dict)
                       else getattr(saved_args, "official_normal_coordinate", None))
        if saved_coord != "erp":
            raise ValueError(
                f"Fine-tuned Normal checkpoint has official_normal_coordinate={saved_coord!r}; "
                "do not resume incompatible world-frame labels. Use --finetune with the PanoMAE pretrained encoder."
            )
        old_source = (saved_args.get("normal_label_source") if isinstance(saved_args, dict)
                      else getattr(saved_args, "normal_label_source", None))
        old_target_frame = (saved_args.get("normal_target_frame") if isinstance(saved_args, dict)
                            else getattr(saved_args, "normal_target_frame", None))
        if old_source not in (None, "normal") or old_target_frame not in (None, "global"):
            raise ValueError(
                "Incompatible Normal checkpoint: previous target labels/frame are "
                f"{old_source!r}/{old_target_frame!r}, but this run requires normal/global. "
                "Start from the matching MAE pretrained encoder with --finetune, "
                "not the old depth-derived Normal --resume checkpoint."
            )

        old_frame = (saved_args.get("normal_prediction_frame")
                     if isinstance(saved_args, dict) else
                     getattr(saved_args, "normal_prediction_frame", None))
        if old_frame is not None and old_frame != args.normal_prediction_frame:
            if not args.allow_frame_override:
                raise ValueError(
                    f"Checkpoint used normal_prediction_frame={old_frame!r} but CLI has "
                    f"{args.normal_prediction_frame!r}. Check the coordinate convention, "
                    "or explicitly pass --allow_frame_override for diagnostics."
                )
            if misc.is_main_process():
                print(f"[FRAME OVERRIDE] Checkpoint {old_frame} -> CLI {args.normal_prediction_frame}")
        elif old_frame is None and misc.is_main_process():
            print("[FRAME INFO] Legacy checkpoint does not record normal_prediction_frame. "
                  "Inspect its training GT convention before selecting legacy_yaw180.")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        if not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            loss_scaler.load_state_dict(checkpoint["scaler"])
            args.start_epoch = checkpoint["epoch"] + 1
    if args.eval:
        test_stats = engine_normal.evaluate(data_loader_val, model, device, criterion, args, epoch=args.epochs - 1)
        if misc.is_main_process(): print(test_stats)
        return
    if misc.is_main_process():
        print(f"Start training Normal Estimation for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = engine_normal.train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args)
        test_stats = engine_normal.evaluate(data_loader_val, model, device, criterion, args, epoch)
        if diagnostic_loader is not None:
            diag_eval_args = copy.copy(args)
            diag_eval_args.epochs = -1
            diag_eval_args.save_normal_vis_every = 0
            diag_eval_args.eval = False
            diag_stats = engine_normal.evaluate(diagnostic_loader, model, device,
                                                criterion, diag_eval_args, epoch)
            print(f"[Diagnostic][Epoch {epoch}] training-area held-in loss="
                  f"{diag_stats['loss']:.6f} mean={diag_stats['mean']:.3f}deg; "
                  f"held-out val loss={test_stats['loss']:.6f} "
                  f"mean={test_stats['mean']:.3f}deg")
        else:
            diag_stats = None
        if misc.is_main_process():
            if epoch % 20 == 0 or epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch}
            if diag_stats is not None:
                log_stats.update({f"diagnostic_train_eval_{k}": v for k, v in diag_stats.items()})
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    if misc.is_main_process():
        print(f"Training time {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
