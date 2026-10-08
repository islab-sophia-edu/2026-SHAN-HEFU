import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from util.datasets_PanoMAE_gctt_noadaptive import PanoramicDataset
from models_PanoMAE_gctt_noadaptive import vit_base_patch16, vit_large_patch16, vit_huge_patch14
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain_PanoMAE_gctt_noadaptive import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Panoramic MAE Pretraining with GCTT + Oriented SPE | NoAdaptive fixed 0.75 mask', add_help=False)

    # Input resolution parameters
    parser.add_argument('--pano_h', default=512, type=int, help='Input panoramic image height, e.g., 512 or 1024')
    parser.add_argument('--pano_w', default=1024, type=int, help='Input panoramic image width, e.g., 1024 or 2048')

    parser.add_argument('--multiscale_sampling', action='store_true', help='Use multiscale sampling data augmentation.')
    parser.set_defaults(multiscale_sampling=False)
    # NoAdaptive ablation: dynamic mask ratio is disabled and ignored.
    parser.add_argument('--dynamic_mask_ratio', action='store_true',
                        help='Deprecated in noadaptive version: ignored; training always uses mask_ratio=0.75.')
    parser.set_defaults(dynamic_mask_ratio=False)
    parser.add_argument('--no_geometric_bias', action='store_false', dest='geometric_bias', help='Disable geometric bias in positional encoding.')
    parser.set_defaults(geometric_bias=True)
    # NoAdaptive ablation: adaptive/content-aware masking is hard-disabled.
    # Kept for old launch-script compatibility; passing it still leaves adaptive masking disabled.
    parser.add_argument('--no_adaptive_masking', action='store_false', dest='adaptive_masking',
                        help='Deprecated in noadaptive version: no effect; adaptive masking is always disabled.')
    parser.set_defaults(adaptive_masking=False)
    parser.add_argument('--angle_jitter_deg', default=5.0, type=float, help='Spherical random crop angle jitter')
    parser.add_argument('--no_color_jitter', action='store_false', dest='use_color_jitter', help='Disable ColorJitter')
    parser.set_defaults(use_color_jitter=True)
    parser.add_argument('--no_blur', action='store_false', dest='use_blur', help='Disable GaussianBlur')
    parser.set_defaults(use_blur=True)

    # --- GCTT dataset/model parameters ---
    parser.add_argument('--no_gctt', action='store_false', dest='use_gctt', help='Disable Gauge-Consistent Tangent Tokenization.')
    parser.set_defaults(use_gctt=True)
    parser.add_argument('--gctt_gauge_jitter_deg', default=30.0, type=float,
                        help='Training-time global per-image tangent-frame gauge jitter in degrees. Eval uses 0.')
    parser.add_argument('--gctt_local_gauge_jitter_deg', default=0.0, type=float,
                        help='Optional per-token local tangent-frame gauge jitter in degrees. Usually keep 0 first.')
    parser.add_argument('--gauge_num_frequencies', default=16, type=int,
                        help='Deprecated: kept for CLI compatibility. psi_i is now absorbed into oriented SPE, not encoded by standalone Fourier GE.')
    parser.add_argument('--gauge_scale_init', default=0.02, type=float,
                        help='Deprecated: kept for CLI compatibility. No standalone gauge positional scale is used.')
    parser.add_argument('--no_gauge_bias', action='store_false', dest='use_gauge_bias',
                        help='Disable oriented-frame relative attention bias term.')
    parser.set_defaults(use_gauge_bias=True)
    parser.add_argument('--gauge_bias_init', default=0.0, type=float,
                        help='Initial beta for gauge-relative attention bias. 0.0 is stable and learnable.')
    # ------------------------------------

    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Fixed mask ratio. NoAdaptive version hard-locks this to 0.75 during train/eval.')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use per-patch normalized pixels as targets')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--no_angular_bias', action='store_false', dest='use_angular_bias', help='Disable angular distance attention bias.')
    parser.set_defaults(use_angular_bias=True)
    parser.add_argument('--angular_bias_init_slope', default=1.0, type=float, help='Initial max slope for ALiBi-style angular bias.')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Absolute learning rate')
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR', help='Base LR: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR', help='Lower LR bound')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='Warmup epochs')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer-wise LR decay')

    # Checkpoint params
    parser.add_argument('--finetune', default='', type=str, help='Finetune from checkpoint')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--data_path', default='./sun360_outdoor/', type=str, help='Path to full panoramic images')
    parser.add_argument('--val_data_path', default=None, type=str, help='Validation dataset path')
    parser.add_argument('--grid_height', default=4, type=int, help='Number of patches in vertical direction')
    parser.add_argument('--output_dir', default='./output_pano_recon', type=str, help='Output path')
    parser.add_argument('--log_dir', default=None, type=str, help='TensorBoard log path')

    # Dataset / runtime parameters
    parser.add_argument('--eval_freq', default=10, type=int, help='Evaluation frequency in epochs')
    parser.add_argument('--device', default='cuda', type=str, help='Device')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory')
    parser.set_defaults(pin_mem=False)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)

    return parser


def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading pre-trained ViT weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model', checkpoint)

    # Remove unsupported / shape-sensitive keys.
    for k in ['pos_embed', 'head.weight', 'head.bias']:
        if k in checkpoint_model:
            print(f"Removing unused key '{k}' from pre-trained checkpoint.")
            del checkpoint_model[k]

    model_state = model.state_dict()

    # The oriented-SPE version changes angle_pos_embed.fourier_weights from
    # [M, 3] to [M, 9] when --use_gctt is enabled. strict=False does not ignore
    # shape mismatches, so remove incompatible tensor keys before loading.
    for k in list(checkpoint_model.keys()):
        if k in model_state and hasattr(checkpoint_model[k], 'shape'):
            if checkpoint_model[k].shape != model_state[k].shape:
                print(f"Removing shape-mismatched key '{k}' "
                      f"(checkpoint: {checkpoint_model[k].shape}, "
                      f"model: {model_state[k].shape}).")
                del checkpoint_model[k]
    pred_w_key = 'decoder_pred.weight'
    pred_b_key = 'decoder_pred.bias'

    if pred_w_key in checkpoint_model and pred_w_key in model_state:
        if checkpoint_model[pred_w_key].shape != model_state[pred_w_key].shape:
            print(f"Shape mismatch for {pred_w_key} "
                  f"(Checkpoint: {checkpoint_model[pred_w_key].shape}, "
                  f"Model: {model_state[pred_w_key].shape}). Removing.")
            del checkpoint_model[pred_w_key]
            if pred_b_key in checkpoint_model:
                del checkpoint_model[pred_b_key]

    # Interpolate patch/view embedding kernels if patch size changed.
    pe_keys = [
        k for k in checkpoint_model.keys()
        if ('patch_embed' in k or 'view_embed' in k) and 'weight' in k and checkpoint_model[k].ndim == 4
    ]

    for ckpt_key in pe_keys:
        model_key = ckpt_key
        if 'patch_embed' in ckpt_key:
            for candidate in [
                ckpt_key.replace('patch_embed', 'view_embed'),
                ckpt_key.replace('patch_embed.proj', 'view_embed.proj.proj'),
            ]:
                if candidate in model_state:
                    model_key = candidate
                    break

        if model_key in model_state:
            ckpt_weight = checkpoint_model[ckpt_key]
            model_weight = model_state[model_key]

            if ckpt_weight.shape != model_weight.shape:
                print(f"Interpolating {ckpt_key}: {ckpt_weight.shape} -> {model_weight.shape}")
                interpolated_weight = F.interpolate(
                    ckpt_weight,
                    size=model_weight.shape[2:],
                    mode='bicubic',
                    align_corners=False,
                )
                if ckpt_key != model_key:
                    del checkpoint_model[ckpt_key]
                checkpoint_model[model_key] = interpolated_weight

                ckpt_bias = ckpt_key.replace('weight', 'bias')
                model_bias = model_key.replace('weight', 'bias')
                if ckpt_bias in checkpoint_model and model_bias in model_state:
                    checkpoint_model[model_bias] = checkpoint_model[ckpt_bias]
                    if ckpt_bias != model_bias:
                        del checkpoint_model[ckpt_bias]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Restored from pre-trained checkpoint: {msg}")


def main(args):
    # Hard lock this ablation: no adaptive masking and no dynamic ratio.
    # This prevents stale CLI flags or copied launch scripts from changing the mask policy.
    args.dynamic_mask_ratio = False
    args.mask_ratio = 0.75
    args.adaptive_masking = False

    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    v_steps = args.grid_height
    dynamic_img_size = args.pano_h // v_steps

    print(f"Input Pano Size: {args.pano_w}x{args.pano_h}")
    print(f"Grid Height: {v_steps}, Calculated Patch Size: {dynamic_img_size}x{dynamic_img_size}")
    print(f"GCTT sampling: {args.use_gctt}, global gauge jitter: {args.gctt_gauge_jitter_deg}, "
          f"local gauge jitter: {args.gctt_local_gauge_jitter_deg}")
    print("Positional encoding: SPE(theta, phi, psi) via oriented spherical frame; "
          "standalone GE(psi) is not used.")
    print("Masking: NoAdaptive random MAE masking, fixed mask_ratio=0.75 for training/evaluation.")

    dataset_train = PanoramicDataset(
        root_dir=args.data_path,
        grid_height=args.grid_height,
        img_size=dynamic_img_size,
        pano_h=args.pano_h,
        pano_w=args.pano_w,
        multiscale_sampling=args.multiscale_sampling,
        use_color_jitter=args.use_color_jitter,
        use_blur=args.use_blur,
        angle_jitter_deg=args.angle_jitter_deg,
        use_gctt=args.use_gctt,
        gctt_gauge_jitter_deg=args.gctt_gauge_jitter_deg,
        gctt_local_gauge_jitter_deg=args.gctt_local_gauge_jitter_deg,
    )
    dataset_val = None
    if args.val_data_path:
        dataset_val = PanoramicDataset(
            root_dir=args.val_data_path,
            grid_height=args.grid_height,
            img_size=dynamic_img_size,
            pano_h=args.pano_h,
            pano_w=args.pano_w,
            is_training=False,
            use_gctt=args.use_gctt,
            gctt_gauge_jitter_deg=0.0,
            gctt_local_gauge_jitter_deg=0.0,
        )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False
        ) if dataset_val else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = None
    if dataset_val:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    model = globals()[args.model](
        img_size=dynamic_img_size,
        norm_pix_loss=args.norm_pix_loss,
        geometric_bias=args.geometric_bias,
        adaptive_masking=False,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        use_gctt=args.use_gctt,
        gauge_num_frequencies=args.gauge_num_frequencies,
        gauge_scale_init=args.gauge_scale_init,
        use_gauge_bias=args.use_gauge_bias,
        gauge_bias_init=args.gauge_bias_init,
    )

    if args.finetune:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model
    print(f"Model: {args.model}, Parameters (M): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6:.2f}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Peak LR for the schedule: {args.lr:.8f}")

    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    len_loader = len(data_loader_train)
    steps_per_epoch = len_loader // args.accum_iter
    if len_loader % args.accum_iter != 0:
        print(f"Warning: len_loader ({len_loader}) is not divisible by accum_iter ({args.accum_iter}).")

    warmup_steps = args.warmup_epochs * steps_per_epoch
    main_steps = (args.epochs - args.warmup_epochs) * steps_per_epoch

    print(f"Scheduler: {steps_per_epoch} updates per epoch. "
          f"Warmup for {warmup_steps} updates, then decay for {main_steps} updates.")

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=main_steps, eta_min=args.min_lr
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        resumed_lr = optimizer.param_groups[0]['lr']
        print(f"Resumed training from epoch {args.start_epoch}. Current LR: {resumed_lr:.8f}")

    log_writer = None
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, scheduler, device, epoch, loss_scaler, args
        )

        test_stats = {}
        if data_loader_val and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            test_stats = evaluate(model, data_loader_val, device, epoch, args)

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                scheduler=scheduler,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer:
                for k, v in log_stats.items():
                    log_writer.add_scalar(k, v, epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == '__main__':
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
