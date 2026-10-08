import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models_depth_pb as models_depth
import engine_depth_pb as engine_depth
from dataset_stanford_depth_pb import build_depth_dataset
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("PB Pano Depth Estimation", add_help=False)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=4, type=int)
    parser.add_argument("--model", default="vit_base_patch16", type=str, metavar="MODEL")

    parser.add_argument("--pano_h", default=2048, type=int)
    parser.add_argument("--pano_w", default=4096, type=int)
    parser.add_argument("--grid_height", default=16, type=int)
    parser.add_argument("--in_chans", default=3, type=int,
                        help="Minimal v7 uses RGB only. rayXYZ/ray-token embedding is disabled; keep this as 3.")

    # Depth normalization / evaluation. The requested cliff is 100m, not 10m.
    parser.add_argument("--max_depth", default=100.0, type=float,
                        help="Depth cliff in meters. Dataset stores targets as depth_meters / max_depth.")
    parser.add_argument("--depth_scale", default=512.0, type=float,
                        help="Stanford raw depth scale: depth_meters = raw / depth_scale.")
    parser.add_argument("--min_depth", default=0.05, type=float,
                        help="Minimum valid metric depth used by log-depth/bin head and loss.")
    parser.add_argument("--depth_valid_min", default=0.05, type=float,
                        help="Raw metric depths below this value are ignored before clipping/normalization.")
    parser.add_argument("--invalid_depth_raw", default=65535, type=int,
                        help="Stanford 2D-3D-S missing depth code. Only this exact raw value is invalid; real far depths are kept and clipped to --max_depth.")
    parser.add_argument("--keep_over_max_depth_as_cliff", action="store_true", default=True,
                        help="Keep valid raw depths above --max_depth and clip them to the cliff instead of deleting them.")
    parser.add_argument("--depth_head_type", default="log", choices=["bins", "log", "sigmoid"],
                        help="Compact depth head. In v5, bins is accepted as a CLI-compatible alias for log.")
    parser.add_argument("--num_depth_bins", default=80, type=int,
                        help="Kept for command compatibility. v5 compact decoder does not use bin logits.")

    # Deep encoder adaptation. This is the main v5 change: keep and load the
    # pretrained PB encoder, then add a memory-light depth-specific encoder
    # before the original compact decoder.
    parser.add_argument("--depth_encoder_depth", default=8, type=int,
                        help="Number of ConvNeXt-style grid-token blocks after the pretrained encoder.")
    parser.add_argument("--stage_encoder_depth", default=1, type=int,
                        help="Number of grid-token blocks applied to each skip feature.")
    parser.add_argument("--depth_encoder_mlp_ratio", default=0.5, type=float)
    parser.add_argument("--depth_encoder_kernel_size", default=3, type=int)
    parser.add_argument("--depth_encoder_drop_path", default=0.0, type=float)
    parser.add_argument("--depth_encoder_init_scale", default=1e-2, type=float,
                        help="Residual scale for new depth encoder blocks. Larger learns faster; smaller preserves pretrained behavior.")
    parser.add_argument("--cnn_base_dim", default=64, type=int,
                        help="Keep 64 for the original small decoder. Lower to 48 if OOM.")
    parser.add_argument("--adapter_lr_mult", default=5.0, type=float,
                        help="Multiply LR for depth_encoder/stage_adapters/depth_head compared with the base LR.")
    parser.add_argument("--no_adapter_lr_mult", action="store_true",
                        help="Disable the special LR multiplier for new depth-specific modules.")

    # Range-aware loss weights for cliff=100m.
    parser.add_argument("--w_silog", default=0.7, type=float)
    parser.add_argument("--w_grad", default=0.2, type=float)
    parser.add_argument("--w_berhu", default=0.2, type=float)
    parser.add_argument("--w_rmse", default=1.0, type=float,
                        help="Direct metric RMSE-style supervision in meters/max_depth.")
    parser.add_argument("--w_far_l1", default=0.5, type=float,
                        help="Far-depth weighted L1 term to reduce large meter errors.")
    parser.add_argument("--w_bin", default=0.0, type=float,
                        help="v5 compact decoder does not emit bin logits; keep this 0.")
    parser.add_argument("--far_loss_boost", default=4.0, type=float,
                        help="Extra weight for far pixels: 1 + boost*(depth/max_depth)^gamma.")
    parser.add_argument("--far_gamma", default=2.0, type=float)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")
    parser.add_argument("--dist_eval", action="store_true", default=False)

    parser.add_argument("--data_path", default="./train", type=str)
    parser.add_argument("--val_data_path", default="./test", type=str)
    parser.add_argument("--train_areas", default="area_1,area_2,area_3,area_4,area_6", type=str)
    parser.add_argument("--val_areas", default="area_5a,area_5b", type=str)
    parser.add_argument("--rgb_folder_name", default="rgb", type=str)
    parser.add_argument("--depth_folder_name", default="depth", type=str)
    parser.add_argument("--debug_limit", default=None, type=int)

    # PB spherical PE options. No gauge/oriented-frame options are exposed in this ablation.
    parser.add_argument("--no_angular_bias", action="store_false", dest="use_angular_bias",
                        help="Disable PB spherical angular attention bias.")
    parser.set_defaults(use_angular_bias=True)
    parser.add_argument("--angular_bias_init_slope", default=1.0, type=float)

    parser.add_argument("--no_geometric_bias", action="store_false", dest="geometric_bias",
                        help="Disable PB stratified geometric Fourier initialization.")
    parser.set_defaults(geometric_bias=True)

    # Train-time augmentations matching your PB pretraining setup. Dataset disables them for val/test.
    parser.add_argument("--multiscale_sampling", action="store_true", help="Use PB multiscale FOV sampling [0.8, 1.2] during training.")
    parser.set_defaults(multiscale_sampling=False)
    parser.add_argument("--angle_jitter_deg", default=5.0, type=float)
    parser.add_argument("--no_color_jitter", action="store_false", dest="use_color_jitter")
    parser.set_defaults(use_color_jitter=True)
    parser.add_argument("--no_blur", action="store_false", dest="use_blur")
    parser.set_defaults(use_blur=True)
    parser.add_argument("--no_full_pose3d", action="store_false", dest="use_full_pose3d")
    parser.set_defaults(use_full_pose3d=True)
    parser.add_argument("--pose_yaw_deg", default=360.0, type=float)
    parser.add_argument("--pose_pitch_deg", default=30.0, type=float)
    parser.add_argument("--pose_roll_deg", default=30.0, type=float)
    parser.add_argument("--use_horizontal_roll", action="store_true", help="Optional PB horizontal roll augmentation.")
    parser.set_defaults(use_horizontal_roll=False)
    parser.add_argument("--hflip_prob", default=0.0, type=float)
    parser.add_argument("--aug_device", default="cuda", type=str)

    # Optional: save validation comparison more frequently. Last epoch is always saved.
    parser.add_argument("--save_depth_vis_every", default=0, type=int)
    parser.add_argument("--inverse_odi_blend", default="average", choices=["overwrite", "average"],
                        help="Inverse ODI visualization blending. v8 defaults to average for scalar depth before colormap; use overwrite for old odi_processing-style debugging.")
    parser.add_argument("--vis_depth_norm", default="dynamic", choices=["dynamic", "fixed"],
                        help="Depth color normalization for visualization only. dynamic maps the current GT panorama percentile to red; fixed uses vis_depth_max/max_depth.")
    parser.add_argument("--vis_depth_min", default=0.0, type=float,
                        help="Visualization near depth in meters. Usually keep 0 so near is blue.")
    parser.add_argument("--vis_depth_max", default=0.0, type=float,
                        help="Visualization far depth in meters for --vis_depth_norm fixed. 0 means use --max_depth. Does not affect metrics/training.")
    parser.add_argument("--vis_depth_percentile", default=99.0, type=float,
                        help="For dynamic visualization, this GT depth percentile is mapped to red. Metrics/training still use --max_depth.")
    parser.add_argument("--vis_error_max", default=0.0, type=float,
                        help="Visualization error scale in meters. 0 means dynamic percentile.")
    parser.add_argument("--vis_error_percentile", default=99.0, type=float,
                        help="For dynamic error visualization, this error percentile is mapped to red.")
    parser.add_argument("--save_patch_grid_vis", action="store_true",
                        help="Also save the old patch-grid visualization for debugging.")

    parser.add_argument("--output_dir", default="./output_depth")
    parser.add_argument("--log_dir", default="./output_depth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--finetune", default="", help="Path to PB MAE pretrained checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def _adapt_view_embed_weight(value, target_shape):
    """
    Adapt pretrained view_embed.proj.proj.weight to a different channel count or patch size.
    Handles 3ch MAE checkpoints and optional future 6ch depth models.
    """
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
        print("============== Loading PB Pretrained Weights ==============")
        print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_dict = model.state_dict()

    new_dict = {}
    loaded_keys = []

    for key, value in state_dict.items():
        k = key[7:] if key.startswith("module.") else key

        # Load the pretrained PB encoder AND MAE decoder blocks.
        # Only task-specific dense heads are skipped.
        if (
            k == "mask_token"
            or k.startswith("decoder_pred")
            or k.startswith("head")
            or k.startswith("depth_head")
            or k.startswith("depth_refine")
            or k.startswith("global_detail")
            or k.startswith("cnn_proj")
            or k.startswith("fusion")  # old AttentionFusion compatibility
            or k.startswith("neck_conv")
            or k.startswith("decoder_neck")
            or k.startswith("skip")
            or k.startswith("up")
        ):
            continue

        # Backward compatibility with old ViT/MAE checkpoints.
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
                    if misc.is_main_process():
                        print(f"  [Skip mismatch] {k}: ckpt {tuple(value.shape)} != model {tuple(target_shape)}")
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

    msg = model.load_state_dict(new_dict, strict=False)

    if misc.is_main_process():
        print(f"[Success] Loaded {len(loaded_keys)} keys.")
        print(msg)



def _boost_task_param_groups(param_groups, model, adapter_lr_mult):
    """
    Split layer-decay groups so new depth-specific modules can learn faster
    while pretrained PB encoder weights still use the normal layer-decay schedule.
    """
    if adapter_lr_mult is None or float(adapter_lr_mult) == 1.0:
        return param_groups

    task_prefixes = (
        "depth_encoder",
        "stage_adapters",
        "depth_head",
        "global_detail",
        "cnn_proj",
        "fusion",  # old AttentionFusion compatibility
        "neck_conv",
        "skip",
        "up",
    )
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    new_groups = []

    for group in param_groups:
        task_params = []
        base_params = []
        for p in group.get("params", []):
            name = id_to_name.get(id(p), "")
            if name.startswith(task_prefixes):
                task_params.append(p)
            else:
                base_params.append(p)

        if base_params:
            g = dict(group)
            g["params"] = base_params
            new_groups.append(g)

        if task_params:
            g = dict(group)
            g["params"] = task_params
            g["lr_scale"] = float(g.get("lr_scale", 1.0)) * float(adapter_lr_mult)
            new_groups.append(g)

    return new_groups

def main(args):
    if torch.cuda.is_available():
        torch.cuda.init()
        if args.device == "cuda":
            torch.cuda.set_device(0)

    misc.init_distributed_mode(args)
    if args.device == "cuda":
        args.device = "cuda:0"
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if misc.is_main_process():
        print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
        print(f"actual lr: {args.lr:.2e}")
        print(f"effective batch size: {eff_batch_size}")
        print(f"Depth cliff / max_depth: {args.max_depth:g} m")
        print(f"Depth validity: raw != {args.invalid_depth_raw}, depth >= {args.depth_valid_min:g}; valid depths above cliff are clipped to {args.max_depth:g}m")
        print(f"Input channels: {args.in_chans} (RGB only; no rayXYZ/ray token)")
        print(f"Depth head: {args.depth_head_type} compact head, min_depth={args.min_depth:g} m")
        print(f"Depth task encoder: depth={args.depth_encoder_depth}, stage_depth={args.stage_encoder_depth}, "
              f"mlp_ratio={args.depth_encoder_mlp_ratio}, kernel={args.depth_encoder_kernel_size}, "
              f"init_scale={args.depth_encoder_init_scale}, adapter_lr_mult={args.adapter_lr_mult if not args.no_adapter_lr_mult else 1.0}")
        print("Decoder fusion: GCTT-style cnn_proj1/2/3 + concat with ViT skips; PB-only geometry is unchanged.")
        print(f"Loss weights: silog={args.w_silog}, grad={args.w_grad}, berhu={args.w_berhu}, rmse={args.w_rmse}, far_l1={args.w_far_l1}, bin={args.w_bin}, far_boost={args.far_loss_boost}")
        print("PB ablation: no gauge_angles, no oriented SPE, no gauge bias.")
        print("Positional encoding: PB AnglePositionalEncoding(theta, phi) on S^2 + optional angular distance bias.")
        print(
            "Train augmentations: "
            f"full_pose3d={args.use_full_pose3d} "
            f"(yaw=[0,{args.pose_yaw_deg}], pitch=±{args.pose_pitch_deg}, roll=±{args.pose_roll_deg}), "
            f"GaussianBlur={args.use_blur}, ColorJitter={args.use_color_jitter}, "
            f"multiscale_sampling={args.multiscale_sampling}, horizontal_roll={args.use_horizontal_roll}, "
            f"angle_jitter_deg={args.angle_jitter_deg}. "
            "Validation disables full_pose3d / blur / color jitter / multiscale / angle jitter by dataset is_train=False."
        )

    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w)
    args.input_size = real_patch_size

    dataset_train = build_depth_dataset(is_train=True, args=args)
    dataset_val = build_depth_dataset(is_train=False, args=args)

    if misc.is_main_process():
        print(f"Train dataset size: {len(dataset_train)}")
        print(f"Val dataset size  : {len(dataset_val)}")

    if len(dataset_train) <= 0:
        raise ValueError("Train dataset is empty. Check --data_path, area split, and pano/rgb + pano/depth folders.")
    if len(dataset_val) <= 0:
        raise ValueError("Validation dataset is empty. Check --val_data_path, area split, and pano/rgb + pano/depth folders.")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print("Warning: distributed evaluation dataset is not divisible by process count.")
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if misc.is_main_process():
        print(f"Creating PB PanoDepth model: {args.model}")

    model = models_depth.__dict__[args.model](
        img_size=real_patch_size,
        patch_size=real_patch_size,
        in_chans=args.in_chans,
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
        geometric_bias=args.geometric_bias,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        depth_head_type=args.depth_head_type,
        num_depth_bins=args.num_depth_bins,
        depth_encoder_depth=args.depth_encoder_depth,
        stage_encoder_depth=args.stage_encoder_depth,
        depth_encoder_mlp_ratio=args.depth_encoder_mlp_ratio,
        depth_encoder_kernel_size=args.depth_encoder_kernel_size,
        depth_encoder_drop_path=args.depth_encoder_drop_path,
        depth_encoder_init_scale=args.depth_encoder_init_scale,
        cnn_base_dim=args.cnn_base_dim,
    )

    if args.finetune:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    if not args.no_adapter_lr_mult:
        param_groups = _boost_task_param_groups(param_groups, model_without_ddp, args.adapter_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = engine_depth.DepthLoss(
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        w_silog=args.w_silog,
        w_berhu=args.w_berhu,
        w_grad=args.w_grad,
        w_rmse=args.w_rmse,
        w_far_l1=args.w_far_l1,
        w_bin=args.w_bin,
        far_loss_boost=args.far_loss_boost,
        far_gamma=args.far_gamma,
    ).to(device)

    if args.resume:
        if misc.is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_scaler.load_state_dict(checkpoint["scaler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        test_stats = engine_depth.evaluate(data_loader_val, model, device, criterion, args, epoch=args.epochs - 1)
        if misc.is_main_process():
            print(test_stats)
        return

    if misc.is_main_process():
        print(f"Start training Depth Estimation for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = engine_depth.train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )
        test_stats = engine_depth.evaluate(data_loader_val, model, device, criterion, args, epoch)

        if misc.is_main_process():
            if epoch % 20 == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if misc.is_main_process():
        print(f"Training time {total_time_str}")


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
