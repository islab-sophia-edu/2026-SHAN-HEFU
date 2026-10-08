import argparse
import os
import time
import json
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import models_segmentation_noadaptive as models_segmentation
import engine_segmentation_noadaptive as engine_segmentation
from datasets_segmentation_8class_noadaptive import build_segmentation_dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd


def get_args_parser():
    parser = argparse.ArgumentParser("Pano Segmentation Finetuning noadaptive", add_help=False)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=4, type=int)
    parser.add_argument("--model", default="vit_base_patch16", type=str, metavar="MODEL")

    # noadaptive compatibility flags. Segmentation uses the full token grid; these
    # are forced below so old commands/checkpoints cannot enable adaptive masking.
    parser.add_argument("--mask_ratio", default=0.75, type=float, help="Compatibility only. Fixed to 0.75 for noadaptive runs.")
    parser.add_argument("--dynamic_mask_ratio", action="store_true", help="Compatibility only. Forced off in noadaptive runs.")
    parser.set_defaults(dynamic_mask_ratio=False)
    parser.add_argument("--adaptive_masking", action="store_true", help="Compatibility only. Forced off in noadaptive runs.")
    parser.add_argument("--no_adaptive_masking", action="store_false", dest="adaptive_masking")
    parser.set_defaults(adaptive_masking=False)

    parser.add_argument("--pano_h", default=2048, type=int)
    parser.add_argument("--pano_w", default=4096, type=int)
    parser.add_argument("--grid_height", default=16, type=int)
    parser.add_argument("--nb_classes", default=8, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")

    # Dataset paths.
    parser.add_argument("--data_path", default="./train", type=str)
    parser.add_argument("--val_data_path", default="./test", type=str)
    parser.add_argument("--rgb_shared_dir", default=None, type=str,
                        help="Optional shared RGB directory. If omitted, each split uses <data_path>/rgb and <val_data_path>/rgb.")
    parser.add_argument("--debug_limit", default=None, type=int)

    # GCTT / oriented SPE options.
    parser.add_argument("--no_gctt", action="store_false", dest="use_gctt",
                        help="Disable Gauge-Consistent Tangent Tokenization.")
    parser.set_defaults(use_gctt=True)
    parser.add_argument("--gctt_gauge_jitter_deg", default=30.0, type=float,
                        help="Training-time global per-image tangent-frame gauge jitter in degrees.")
    parser.add_argument("--gctt_local_gauge_jitter_deg", default=0.0, type=float,
                        help="Optional per-token local tangent-frame gauge jitter in degrees.")
    parser.add_argument("--no_gauge_bias", action="store_false", dest="use_gauge_bias",
                        help="Disable oriented-frame relative attention bias.")
    parser.set_defaults(use_gauge_bias=True)
    parser.add_argument("--gauge_bias_init", default=0.0, type=float)

    parser.add_argument("--no_angular_bias", action="store_false", dest="use_angular_bias",
                        help="Disable spherical angular attention bias.")
    parser.set_defaults(use_angular_bias=True)
    parser.add_argument("--angular_bias_init_slope", default=1.0, type=float)

    parser.add_argument("--no_geometric_bias", action="store_false", dest="geometric_bias",
                        help="Disable stratified geometric Fourier initialization.")
    parser.set_defaults(geometric_bias=True)
    parser.add_argument("--angle_jitter_deg", default=0.0, type=float,
                        help="Optional spherical center angle jitter for segmentation patches.")
    parser.add_argument("--no_color_jitter", action="store_false", dest="use_color_jitter",
                        help="Disable pretraining-style ColorJitter for training.")
    parser.set_defaults(use_color_jitter=True)
    parser.add_argument("--no_blur", action="store_false", dest="use_blur",
                        help="Disable pretraining-style GaussianBlur for training.")
    parser.set_defaults(use_blur=True)
    parser.add_argument("--no_full_pose3d", action="store_false", dest="use_full_pose3d",
                        help="Disable pretraining-style yaw/pitch/roll ERP rotation for training.")
    parser.set_defaults(use_full_pose3d=True)
    parser.add_argument("--pose_yaw_deg", default=360.0, type=float,
                        help="Training ERP yaw rotation range: uniform [0, value].")
    parser.add_argument("--pose_pitch_deg", default=30.0, type=float,
                        help="Training ERP pitch rotation range: uniform [-value, value].")
    parser.add_argument("--pose_roll_deg", default=30.0, type=float,
                        help="Training ERP roll rotation range: uniform [-value, value].")
    parser.add_argument("--no_horizontal_roll", action="store_false", dest="use_horizontal_roll",
                        help="Disable extra horizontal roll augmentation for training.")
    parser.set_defaults(use_horizontal_roll=True)
    parser.add_argument("--aug_device", default="cuda", type=str)

    parser.add_argument("--output_dir", default="./output_seg_noadaptive")
    parser.add_argument("--log_dir", default="./output_seg_noadaptive")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--finetune", default="", help="Path to MAE pretrained checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def load_pretrained_weights(model, checkpoint_path):
    if misc.is_main_process():
        print("============== Loading Pretrained Weights ==============")
        print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_dict = model.state_dict()

    new_dict = {}
    loaded_keys = []

    for key, value in state_dict.items():
        k = key[7:] if key.startswith("module.") else key

        # Segmentation uses encoder weights only.
        if (
            k.startswith("decoder")
            or k == "mask_token"
            or k.startswith("head")
            or k.startswith("global_detail")
            or k.startswith("cnn_proj")
            or k.startswith("neck_conv")
            or k.startswith("skip")
            or k.startswith("up")
        ):
            continue

        # Backward compatibility: old ViT checkpoints may use patch_embed.
        if "patch_embed.proj" in k:
            k = k.replace("patch_embed.proj", "view_embed.proj.proj")
        elif k.startswith("patch_embed"):
            k = k.replace("patch_embed", "view_embed")

        if k not in model_dict:
            continue

        target_shape = model_dict[k].shape
        if hasattr(value, "shape") and value.shape != target_shape:
            if k == "view_embed.proj.proj.weight" and value.dim() == 4:
                if misc.is_main_process():
                    print(f"  [Interpolate] {k}: {tuple(value.shape)} -> {tuple(target_shape)}")
                value = F.interpolate(
                    value,
                    size=target_shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
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


def main(args):
    # Hard noadaptive setting for this variant.
    # Segmentation itself has no MAE masking path, but keeping these attributes
    # fixed prevents old launch scripts from re-enabling adaptive/dynamic masking.
    args.dynamic_mask_ratio = False
    args.adaptive_masking = False
    args.mask_ratio = 0.75

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
        print(f"GCTT: {args.use_gctt}, global gauge jitter: {args.gctt_gauge_jitter_deg}, "
              f"local gauge jitter: {args.gctt_local_gauge_jitter_deg}")
        print("Positional encoding: SPE(theta, phi, psi) via oriented spherical frame; "
              "standalone GE(psi) is not used.")
        print(
            "Train augmentations: "
            f"full_pose3d={args.use_full_pose3d} "
            f"(yaw=[0,{args.pose_yaw_deg}], pitch=±{args.pose_pitch_deg}, roll=±{args.pose_roll_deg}), "
            f"GaussianBlur={args.use_blur}, ColorJitter={args.use_color_jitter}, "
            f"horizontal_roll={args.use_horizontal_roll}, angle_jitter_deg={args.angle_jitter_deg}. "
            "Validation disables full_pose3d / blur / color jitter / gauge jitter / angle jitter by dataset is_train=False."
        )

    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w)
    args.input_size = real_patch_size

    dataset_train = build_segmentation_dataset(is_train=True, args=args)
    dataset_val = build_segmentation_dataset(is_train=False, args=args)

    if misc.is_main_process():
        print(f"Train dataset size: {len(dataset_train)}")
        print(f"Val dataset size  : {len(dataset_val)}")

    if len(dataset_train) <= 0:
        raise ValueError(
            "Train dataset is empty. Check --data_path, the mask directory, "
            "and the rgb directory under the split, or pass --rgb_shared_dir explicitly."
        )
    if len(dataset_val) <= 0:
        raise ValueError(
            "Validation dataset is empty. Check --val_data_path, the mask directory, "
            "and the rgb directory under the split, or pass --rgb_shared_dir explicitly."
        )

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
        print(f"Creating model: {args.model} with encoder + 2D CNN segmentation decoder")

    model = models_segmentation.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=real_patch_size,
        patch_size=real_patch_size,
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
        geometric_bias=args.geometric_bias,
        use_gctt=args.use_gctt,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        use_gauge_bias=args.use_gauge_bias,
        gauge_bias_init=args.gauge_bias_init,
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
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    if args.eval:
        test_stats = engine_segmentation.evaluate(data_loader_val, model, device, criterion, args, epoch=args.start_epoch)
        print(f"Eval stats: {test_stats}")
        return

    if misc.is_main_process():
        print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = engine_segmentation.train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )

        test_stats = engine_segmentation.evaluate(data_loader_val, model, device, criterion, args, epoch)

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
