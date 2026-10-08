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

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from PanoMAE_classification_dataset_noadaptive import build_classification_dataset
from models_PanoMAE_classification_noadaptive import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from engine_PanoMAE_classification_noadaptive import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser("PanoMAE Classification with GCTT + Oriented SPE [NoAdaptive]", add_help=False)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    # Panorama / tangent-view geometry
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

    # GCTT / oriented SPE
    parser.add_argument("--use_gctt", action="store_true", dest="use_gctt")
    parser.add_argument("--no_gctt", action="store_false", dest="use_gctt")
    parser.set_defaults(use_gctt=True)
    parser.add_argument("--no_geometric_bias", action="store_false", dest="geometric_bias")
    parser.set_defaults(geometric_bias=True)
    parser.add_argument("--no_angular_bias", action="store_false", dest="use_angular_bias")
    parser.set_defaults(use_angular_bias=True)
    parser.add_argument("--angular_bias_init_slope", default=1.0, type=float)
    parser.add_argument("--no_gauge_bias", action="store_false", dest="use_gauge_bias")
    parser.set_defaults(use_gauge_bias=True)
    parser.add_argument("--gauge_bias_init", default=0.0, type=float)

    # Deprecated but kept for compatibility with old command lines/checkpoints.
    parser.add_argument("--gauge_num_frequencies", default=16, type=int)
    parser.add_argument("--gauge_scale_init", default=0.02, type=float)

    # NoAdaptive compatibility. Classification itself has no MAE masking path;
    # these args are accepted only so shared noadaptive/pretrain command lines do
    # not fail, and are force-disabled in main().
    parser.add_argument("--dynamic_mask_ratio", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--adaptive_masking", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_adaptive_masking", action="store_false", dest="adaptive_masking", help=argparse.SUPPRESS)
    parser.set_defaults(dynamic_mask_ratio=False, adaptive_masking=False)

    # Data augmentation
    parser.add_argument("--multiscale_sampling", action="store_true")
    parser.set_defaults(multiscale_sampling=False)
    parser.add_argument("--angle_jitter_deg", default=5.0, type=float)
    parser.add_argument("--use_full_pose3d", action="store_true")
    parser.set_defaults(use_full_pose3d=True)
    parser.add_argument("--no_full_pose3d", action="store_false", dest="use_full_pose3d")
    parser.add_argument("--use_horizontal_roll", action="store_true")
    parser.set_defaults(use_horizontal_roll=False)
    parser.add_argument("--no_color_jitter", action="store_false", dest="use_color_jitter")
    parser.set_defaults(use_color_jitter=True)
    parser.add_argument("--no_blur", action="store_false", dest="use_blur")
    parser.set_defaults(use_blur=True)
    parser.add_argument("--gctt_gauge_jitter_deg", default=30.0, type=float)
    parser.add_argument("--gctt_local_gauge_jitter_deg", default=0.0, type=float)
    parser.add_argument(
        "--aug_device",
        default="cpu",
        type=str,
        help="Use 'cpu' for stable DataLoader workers. Use 'cuda' only if your loader setup supports GPU augmentation.",
    )

    # Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Label augmentation
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--recount", type=int, default=1)

    # IO / runtime
    parser.add_argument("--finetune", default="")
    parser.add_argument("--data_path", default="./sun360/train", type=str)
    parser.add_argument("--val_data_path", default="./sun360/val", type=str)
    parser.add_argument("--output_dir", default="./output_dir_cls")
    parser.add_argument("--log_dir", default="./output_dir_cls")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")

        # Classification fine-tuning does not load MAE decoder/mask or an old classifier head.
        if (
            k.startswith("head")
            or k.startswith("decoder_")
            or k.startswith("decoder.")
            or k.startswith("decoder_pos_embed")
            or k.startswith("decoder_blocks")
            or k == "mask_token"
        ):
            continue

        # Old pretraining used enc_ang_bias/dec_ang_bias; classifier uses ang_bias.
        if k.startswith("enc_ang_bias."):
            k = k.replace("enc_ang_bias.", "ang_bias.", 1)
        if k.startswith("dec_ang_bias."):
            continue

        if k not in model_state:
            continue

        if hasattr(v, "shape") and v.shape != model_state[k].shape:
            if ("view_embed" in k or "patch_embed" in k) and "weight" in k and v.ndim == 4:
                print(f"Interpolating {k}: {v.shape} -> {model_state[k].shape}")
                v = F.interpolate(
                    v,
                    size=model_state[k].shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                print(f"Skipping shape-mismatched key {k}: {v.shape} vs {model_state[k].shape}")
                continue

        new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)

    missing_keys = msg.missing_keys
    critical_missing = [
        k for k in missing_keys
        if k.startswith("view_embed") or k.startswith("blocks.0.")
    ]
    if len(critical_missing) > 0:
        print("\n" + "=" * 50)
        print("[CRITICAL WARNING] Some backbone keys were not loaded.")
        print(f"Sample: {critical_missing[:8]}")
        print("This can be normal only when the checkpoint architecture is intentionally different.")
        print("=" * 50 + "\n")

    print(f"Loaded keys: {len(new_state_dict)}")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    return msg


def save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, max_accuracy, tag):
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
    # NoAdaptive fine-tuning path: adaptive masking / dynamic masking are
    # pretraining-only mechanisms, so they are force-disabled here.
    args.dynamic_mask_ratio = False
    args.adaptive_masking = False

    misc.init_distributed_mode(args)
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dynamic_img_size = args.pano_h // args.grid_height
    print(f"Input panorama: {args.pano_w}x{args.pano_h}")
    print(f"Grid height: {args.grid_height}; tangent patch size: {dynamic_img_size}")
    print(
        f"GCTT: {args.use_gctt}; global gauge jitter: {args.gctt_gauge_jitter_deg}; "
        f"local gauge jitter: {args.gctt_local_gauge_jitter_deg}"
    )
    print("Positional encoding: SPE(theta, phi, psi) via oriented local spherical frame; no standalone GE(psi).")
    print("NoAdaptive mode: classification has no MAE masking path; adaptive_masking=False, dynamic_mask_ratio=False.")

    print("Building Training Dataset...")
    dataset_train = build_classification_dataset(is_train=True, args=args)

    train_mapping = dataset_train.class_to_idx
    print(f"Training class mapping: {len(train_mapping)} classes (others excluded).")

    print("Building Validation Dataset...")
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
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup/CutMix is activated.")
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
        geometric_bias=args.geometric_bias,
        use_gctt=args.use_gctt,
        gauge_num_frequencies=args.gauge_num_frequencies,
        gauge_scale_init=args.gauge_scale_init,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        use_gauge_bias=args.use_gauge_bias,
        gauge_bias_init=args.gauge_bias_init,
    )

    if args.finetune and not args.eval:
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

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Peak LR: {args.lr:.8f}")

    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

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
        print(f"Resuming from {args.resume}")
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

    log_writer = None
    if misc.is_main_process() and args.log_dir:
        try:
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        except Exception:
            pass

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

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
                print(f">> New Best Acc1: {max_accuracy:.2f}% saved.")

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
            except Exception:
                pass

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            try:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
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
