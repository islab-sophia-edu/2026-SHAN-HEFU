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
from torch.utils.tensorboard import SummaryWriter

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from PanoMAE_classification_dataset_2dpe import build_classification_dataset
from models_PanoMAE_classification_2dpe import (
    vit_base_patch16,
    vit_large_patch16,
    vit_huge_patch14,
)
from engine_PanoMAE_classification_2dpe import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser("PanoMAE Classification 2DPE Ablation", add_help=False)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    # Input resolution. Same dynamic-size rule as the current non-GCTT PanoMAE.
    parser.add_argument("--pano_h", default=512, type=int)
    parser.add_argument("--pano_w", default=1024, type=int)
    parser.add_argument("--grid_height", default=4, type=int)

    # Model parameters
    parser.add_argument("--model", default="vit_base_patch16", type=str, metavar="MODEL")
    parser.add_argument("--nb_classes", default=32, type=int)
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool")

    parser.add_argument(
        "--no_geometric_bias",
        action="store_false",
        dest="geometric_bias",
        help="Ignored in 2DPE ablation; kept only for CLI/checkpoint compatibility.",
    )
    parser.set_defaults(geometric_bias=True)

    # Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Augmentation
    parser.add_argument("--mixup", type=float, default=0.8, help="mixup alpha")
    parser.add_argument("--cutmix", type=float, default=1.0, help="cutmix alpha")
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--recount", type=int, default=1)

    # IO
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--data_path", default="./sun360/train", type=str)
    parser.add_argument("--val_data_path", default="./sun360/val", type=str)
    parser.add_argument("--output_dir", default="./output_dir_cls_2dpe", type=str)
    parser.add_argument("--log_dir", default="./output_dir_cls_2dpe", type=str)

    # System
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=False)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "encoder.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _map_patch_embed_key(key: str, model_state):
    """
    Allow loading official ViT / MAE checkpoints whose patch embedding is named
    patch_embed.* instead of view_embed.*.
    """
    if "patch_embed" not in key:
        return key

    candidates = [
        key.replace("patch_embed.proj", "view_embed.proj.proj"),
        key.replace("patch_embed", "view_embed"),
    ]

    for candidate in candidates:
        if candidate in model_state:
            return candidate

    return key


def load_pretrained_weights(model, checkpoint_path):
    """
    Load weights from a PanoMAE pretraining checkpoint for clean 2DPE classification ablation.

    Expected to load:
        view_embed.*
        angle_pos_embed.pos_embed only if checkpoint is already 2DPE and shape matches
        blocks.*
        norm.*

    Expected to skip:
        decoder_*
        decoder_blocks.*
        decoder_pos_embed.*
        decoder_norm.*
        decoder_pred.*
        dec_ang_bias.*
        enc_ang_bias.*
        mask_token
        head.*
    """
    print(f"Loading pre-trained weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    filtered_state = {}

    skip_prefixes = (
        "decoder_",
        "decoder_blocks.",
        "decoder_pos_embed.",
        "decoder_norm.",
        "decoder_pred.",
        "dec_ang_bias.",
        "enc_ang_bias.",
        "head.",
    )
    skip_exact = {
        "mask_token",
    }

    skipped_not_found = []
    skipped_shape = []

    for raw_key, value in checkpoint_model.items():
        key = _strip_prefix(raw_key)

        if key in skip_exact or key.startswith(skip_prefixes):
            continue

        # Clean 2DPE classifier does not load learnable 3DPE/SPE Fourier/MLP keys.
        # If the checkpoint is from 2DPE pretraining, angle_pos_embed.pos_embed can load.
        if key.startswith("angle_pos_embed.") and key != "angle_pos_embed.pos_embed":
            continue

        key = _map_patch_embed_key(key, model_state)

        if key not in model_state:
            skipped_not_found.append(key)
            continue

        target_shape = model_state[key].shape

        if value.shape != target_shape:
            if key.startswith("view_embed") and key.endswith("weight") and value.ndim == 4:
                print(f"Interpolating {key}: {tuple(value.shape)} -> {tuple(target_shape)}")
                value = F.interpolate(
                    value,
                    size=target_shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )

                if value.shape != target_shape:
                    skipped_shape.append((key, tuple(value.shape), tuple(target_shape)))
                    continue
            else:
                skipped_shape.append((key, tuple(value.shape), tuple(target_shape)))
                continue

        filtered_state[key] = value

    msg = model.load_state_dict(filtered_state, strict=False)

    # Fail only if the real backbone did not load. Newly added head is expected missing.
    loaded_keys = set(filtered_state.keys())
    critical_keys = [
        "view_embed.proj.proj.weight",
        "blocks.0.attn.qkv.weight",
    ]
    missing_critical = [k for k in critical_keys if k in model_state and k not in loaded_keys]

    if len(missing_critical) > 0:
        print("\n" + "=" * 80)
        print("[CRITICAL ERROR] Backbone weights were not loaded correctly.")
        print(f"Missing critical keys: {missing_critical}")
        print("Check whether the checkpoint is from the current non-GCTT PanoMAE baseline.")
        print("=" * 80 + "\n")
        raise RuntimeError("Critical backbone weights are missing.")

    print(f"[SUCCESS] Loaded {len(filtered_state)} keys from checkpoint.")
    print(f"Missing keys after load: {len(msg.missing_keys)}")
    print(f"Unexpected keys after load: {len(msg.unexpected_keys)}")

    if len(skipped_shape) > 0:
        print(f"[INFO] Shape-mismatched keys skipped: {len(skipped_shape)}")
        for item in skipped_shape[:10]:
            print(f"  - {item[0]}: checkpoint {item[1]} vs model {item[2]}")

    if len(skipped_not_found) > 0:
        print(f"[INFO] Keys not used by classifier: {len(skipped_not_found)}")


def save_checkpoint(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    max_accuracy,
    tag,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint-{tag}.pth"

    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "max_accuracy": max_accuracy,
        "args": args,
    }

    misc.save_on_master(to_save, checkpoint_path)


def main(args):
    misc.init_distributed_mode(args)
    print("{}".format(args).replace(", ", ",\n"))

    if args.pano_h % args.grid_height != 0:
        raise ValueError(
            f"pano_h ({args.pano_h}) must be divisible by grid_height ({args.grid_height})."
        )

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dynamic_img_size = args.pano_h // args.grid_height
    print(f"Input pano size: {args.pano_w}x{args.pano_h}")
    print(f"Grid height: {args.grid_height}, grid width: {2 * args.grid_height}")
    print(f"Patch/view token size: {dynamic_img_size}x{dynamic_img_size}")
    print("Positional Encoding: fixed planar 2D sine-cosine PE; angles are ignored by PE.")

    print("Building training dataset...")
    dataset_train = build_classification_dataset(is_train=True, args=args)

    train_mapping = dataset_train.class_to_idx
    print(f"Training class mapping: {len(train_mapping)} classes.")

    print("Building validation dataset...")
    dataset_val = build_classification_dataset(
        is_train=False,
        args=args,
        class_to_idx=train_mapping,
    )

    if args.nb_classes != len(train_mapping):
        print(f"Info: Updating nb_classes from {args.nb_classes} to {len(train_mapping)}")
        args.nb_classes = len(train_mapping)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )

        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=misc.get_world_size(),
                rank=misc.get_rank(),
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

    mixup_fn = None
    mixup_active = (
        args.mixup > 0.0
        or args.cutmix > 0.0
        or args.cutmix_minmax is not None
    )
    if mixup_active:
        print("Mixup/CutMix is activated. Pano views will be reshaped safely in engine.")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = globals()[args.model](
        img_size=dynamic_img_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        grid_height=args.grid_height,
        geometric_bias=args.geometric_bias,  # ignored by 2DPE model; CLI compatibility only
    )

    if args.finetune and not args.eval:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model

    print(
        f"Model: {args.model}, trainable params (M): "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}"
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.lr:.8f}")
    print(f"Actual LR: {args.lr:.8f}")
    print(f"Effective batch size: {eff_batch_size}")

    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    print("Applying layer-wise learning-rate decay...")
    for group in optimizer.param_groups:
        if "lr_scale" in group:
            group["lr"] = group["lr"] * group["lr_scale"]

    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print(f"Criterion: {criterion}")

    max_accuracy = 0.0

    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

        model_without_ddp.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            loss_scaler.load_state_dict(checkpoint["scaler"])

        if "epoch" in checkpoint and isinstance(checkpoint["epoch"], int):
            args.start_epoch = checkpoint["epoch"] + 1

        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy: {test_stats['acc1']:.2f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    log_writer = None
    if misc.is_main_process() and args.log_dir:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        except Exception:
            pass

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=args.clip_grad,
            mixup_fn=mixup_fn,
            log_writer=log_writer,
            args=args,
        )

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy: {test_stats['acc1']:.2f}%")

        if args.output_dir:
            save_checkpoint(
                args,
                epoch,
                model,
                model_without_ddp,
                optimizer,
                loss_scaler,
                max_accuracy,
                tag="last",
            )

            if test_stats["acc1"] > max_accuracy:
                max_accuracy = test_stats["acc1"]
                save_checkpoint(
                    args,
                    epoch,
                    model,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    max_accuracy,
                    tag="best",
                )
                print(f">> New best Acc1: {max_accuracy:.2f}%")

            if epoch % 20 == 0 or epoch + 1 == args.epochs:
                save_checkpoint(
                    args,
                    epoch,
                    model,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    max_accuracy,
                    tag=f"{epoch:04d}",
                )

        if log_writer:
            try:
                log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
                log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
                log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
                log_writer.flush()
            except Exception:
                pass

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            try:
                with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")
            except Exception:
                pass

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")


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
