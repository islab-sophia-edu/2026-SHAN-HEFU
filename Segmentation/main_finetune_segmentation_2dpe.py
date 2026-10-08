import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import models_segmentation_2dpe as models_segmentation
import engine_segmentation_2dpe as engine_segmentation
from datasets_segmentation_8class_2dpe import build_segmentation_dataset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd


def get_args_parser():
    parser = argparse.ArgumentParser("Pano Segmentation Finetuning - clean 2DPE ablation", add_help=False)

    # Basic training
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=4, type=int)
    parser.add_argument("--model", default="vit_base_patch16", type=str, metavar="MODEL")

    # Panorama/grid
    parser.add_argument("--pano_h", default=2048, type=int)
    parser.add_argument("--pano_w", default=4096, type=int)
    parser.add_argument("--grid_height", default=16, type=int)
    parser.add_argument("--nb_classes", default=8, type=int)

    # PanoMAE-style augmentation. These are applied only to train dataset.
    parser.add_argument("--multiscale_sampling", action="store_true", dest="multiscale_sampling")
    parser.add_argument("--no_multiscale_sampling", action="store_false", dest="multiscale_sampling")
    parser.set_defaults(multiscale_sampling=True)

    parser.add_argument("--use_full_pose3d", action="store_true", dest="use_full_pose3d")
    parser.add_argument("--no_full_pose3d", action="store_false", dest="use_full_pose3d")
    parser.set_defaults(use_full_pose3d=True)

    parser.add_argument("--use_horizontal_roll", action="store_true", default=False)
    parser.add_argument("--angle_jitter_deg", default=5.0, type=float)

    parser.add_argument("--use_color_jitter", action="store_true", dest="use_color_jitter")
    parser.add_argument("--no_color_jitter", action="store_false", dest="use_color_jitter")
    parser.set_defaults(use_color_jitter=True)

    parser.add_argument("--use_blur", action="store_true", dest="use_blur")
    parser.add_argument("--no_blur", action="store_false", dest="use_blur")
    parser.set_defaults(use_blur=True)

    parser.add_argument("--aug_device", default="cuda", type=str, help="Device for ERP rotation/patch extraction inside dataset.")

    # Model compatibility with 2DPE ablation
    parser.add_argument("--no_geometric_bias", action="store_false", dest="geometric_bias", help="Ignored in 2DPE ablation; kept for CLI compatibility.")
    parser.set_defaults(geometric_bias=True)
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")

    # Optimizer
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")

    # Dataset/path
    parser.add_argument("--data_path", default="./train", type=str)
    parser.add_argument("--val_data_path", default="./test", type=str)
    parser.add_argument("--rgb_folder", default="rgb", type=str, help="RGB folder name inside each split directory, e.g. train/rgb and test/rgb")
    parser.add_argument("--mask_folder", default="mask", type=str, help="Mask folder name inside each split directory, e.g. train/mask and test/mask")
    parser.add_argument("--debug_limit", default=None, type=int)

    # Output/runtime
    parser.add_argument("--output_dir", default="./output_seg_2dpe")
    parser.add_argument("--log_dir", default="./output_seg_2dpe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--finetune", default="", help="Path to 2DPE/PanoMAE pretrained checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--save_val_visualization", action="store_true")

    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=False)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_eval", action="store_true", default=False)

    return parser


def _candidate_model_keys(ckpt_key):
    """
    3DPE/PanoMAE checkpoint usually uses view_embed.proj.proj.*.
    Older ViT checkpoints may use patch_embed.*; map only in that direction.
    """
    candidates = [ckpt_key]

    if ckpt_key.startswith("patch_embed"):
        candidates.append(ckpt_key.replace("patch_embed", "view_embed"))
        candidates.append(ckpt_key.replace("patch_embed.proj", "view_embed.proj.proj"))

    if ckpt_key.startswith("view_embed.proj.weight"):
        candidates.append(ckpt_key.replace("view_embed.proj.weight", "view_embed.proj.proj.weight"))
    if ckpt_key.startswith("view_embed.proj.bias"):
        candidates.append(ckpt_key.replace("view_embed.proj.bias", "view_embed.proj.proj.bias"))

    # Preserve order while deduplicating.
    out = []
    seen = set()
    for k in candidates:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def load_pretrained_weights(model, checkpoint_path):
    if misc.is_main_process():
        print("============== Loading 2DPE/PanoMAE Pretrained Weights ==============")
        print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_dict = model.state_dict()

    new_dict = {}
    loaded_keys = []
    skipped_keys = []

    for raw_k, v in state_dict.items():
        k = raw_k[7:] if raw_k.startswith("module.") else raw_k

        # Segmentation keeps only the encoder/tokenizer side; 2DPE is fixed and non-learnable.
        if (
            k.startswith("decoder")
            or k.startswith("mask_token")
            or k.startswith("decoder_pos_embed")
            or k.startswith("dec_ang_bias")
            or k.startswith("enc_ang_bias")
            or k.startswith("decoder_blocks")
            or k.startswith("decoder_norm")
            or k.startswith("decoder_pred")
            or k.startswith("head")
            or k in {"pos_embed", "head.weight", "head.bias"}
        ):
            continue

        # Clean 2DPE segmentation does not load learnable 3DPE/SPE Fourier/MLP keys.
        # If the checkpoint is from 2DPE pretraining, angle_pos_embed.pos_embed can load.
        if k.startswith("angle_pos_embed.") and k != "angle_pos_embed.pos_embed":
            continue

        model_key = None
        for cand in _candidate_model_keys(k):
            if cand in model_dict:
                model_key = cand
                break

        if model_key is None:
            skipped_keys.append(k)
            continue

        target_shape = model_dict[model_key].shape
        if v.shape != target_shape:
            # Interpolate patch/view embedding when pretraining patch size differs
            # from segmentation patch size.
            if ("view_embed.proj.proj.weight" in model_key or "patch_embed.proj.weight" in model_key) and v.dim() == 4:
                if misc.is_main_process():
                    print(f"  [Interpolate] {k} -> {model_key}: {tuple(v.shape)} -> {tuple(target_shape)}")
                v = F.interpolate(v, size=target_shape[2:], mode="bicubic", align_corners=False)
            else:
                if misc.is_main_process():
                    print(f"  [Mismatch] {k} -> {model_key}: {tuple(v.shape)} != {tuple(target_shape)}")
                skipped_keys.append(k)
                continue

        new_dict[model_key] = v
        loaded_keys.append(model_key)

    msg = model.load_state_dict(new_dict, strict=False)
    if misc.is_main_process():
        print(f"[Success] Loaded {len(loaded_keys)} encoder-side keys.")
        print(f"[Missing/Unexpected] {msg}")
        if skipped_keys:
            print(f"[Info] Skipped {len(skipped_keys)} non-matching keys.")


def main(args):
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

    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w)
    args.input_size = real_patch_size

    if misc.is_main_process():
        print(f"Input Pano Size: {args.pano_w}x{args.pano_h}")
        print(f"Grid: {v_steps}x{u_steps}; patch size: {real_patch_size}")
        print("Positional Encoding: fixed planar 2D sine-cosine PE; angles are ignored by PE.")
        print(
            "Train augmentations: "
            f"full_pose3d={args.use_full_pose3d}, multiscale={args.multiscale_sampling}, "
            f"color_jitter={args.use_color_jitter}, blur={args.use_blur}, "
            f"angle_jitter_deg={args.angle_jitter_deg}; eval augmentations disabled by dataset."
        )

    dataset_train = build_segmentation_dataset(is_train=True, args=args)
    dataset_val = build_segmentation_dataset(is_train=False, args=args)

    if len(dataset_train) <= 0:
        raise ValueError(
            f"Training dataset is empty. Check --data_path={args.data_path}, "
            f"--rgb_folder={args.rgb_folder}, --mask_folder={args.mask_folder}. "
            "Expected local layout: train/rgb and train/mask."
        )
    if len(dataset_val) <= 0:
        raise ValueError(
            f"Validation dataset is empty. Check --val_data_path={args.val_data_path}, "
            f"--rgb_folder={args.rgb_folder}, --mask_folder={args.mask_folder}. "
            "Expected local layout: test/rgb and test/mask."
        )

    if misc.is_main_process():
        print(f"Dataset sizes: train={len(dataset_train)}, val={len(dataset_val)}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0 and misc.is_main_process():
                print("Warning: distributed eval dataset length is not divisible by number of processes.")
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
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
        print(f"Creating model: {args.model}")

    model = models_segmentation.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=real_patch_size,
        patch_size=real_patch_size,
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
        geometric_bias=args.geometric_bias,
    )

    if args.finetune:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
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

    if args.resume:
        if misc.is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_scaler.load_state_dict(checkpoint["scaler"])
        args.start_epoch = checkpoint["epoch"] + 1

    log_writer = None
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    if args.eval:
        test_stats = engine_segmentation.evaluate(
            data_loader_val,
            model,
            device,
            criterion,
            args,
            epoch=args.start_epoch,
        )
        if misc.is_main_process():
            print(test_stats)
        return

    if misc.is_main_process():
        print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = engine_segmentation.train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args,
        )

        test_stats = {}
        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            test_stats = engine_segmentation.evaluate(
                data_loader_val,
                model,
                device,
                criterion,
                args,
                epoch,
            )

        if args.output_dir and misc.is_main_process():
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
            if log_writer is not None:
                for k, v in log_stats.items():
                    log_writer.add_scalar(k, v, epoch)
                log_writer.flush()
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
