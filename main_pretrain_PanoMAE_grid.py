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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from util.datasets_PanoMAE_grid import PanoramicDataset
from models_PanoMAE_grid import vit_base_patch16, vit_large_patch16, vit_huge_patch14
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain_PanoMAE_grid import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Panoramic MAE Pretraining with pure ERP-grid tokenization',
        add_help=False,
    )

    # Input / grid parameters.
    parser.add_argument('--pano_h', default=512, type=int,
                        help='Input ERP height, e.g. 512 or 1024.')
    parser.add_argument('--pano_w', default=1024, type=int,
                        help='Input ERP width, e.g. 1024 or 2048. Expected ratio is 2:1.')
    parser.add_argument('--grid_height', default=4, type=int,
                        help='Number of vertical grid patches. Horizontal patches are 2*grid_height.')

    # Grid-ablation compatibility flags. These are intentionally inert for tokenization.
    parser.add_argument('--multiscale_sampling', action='store_true',
                        help='Ignored in pure grid mode; kept for command compatibility.')
    parser.set_defaults(multiscale_sampling=False)
    parser.add_argument('--angle_jitter_deg', default=0.0, type=float,
                        help='Ignored in pure grid mode; patch centers are fixed MAE-style grid centers.')

    # Optional full-ERP augmentations. They do not change grid tokenization.
    parser.add_argument('--use_full_pose3d', action='store_true',
                        help='Optional SO(3) rotation augmentation on the full ERP before grid patchify.')
    parser.set_defaults(use_full_pose3d=False)
    parser.add_argument('--use_horizontal_roll', action='store_true',
                        help='Optional horizontal circular shift on the full ERP before grid patchify.')
    parser.set_defaults(use_horizontal_roll=False)
    parser.add_argument('--no_color_jitter', action='store_false', dest='use_color_jitter',
                        help='Disable ColorJitter.')
    parser.set_defaults(use_color_jitter=True)
    parser.add_argument('--no_blur', action='store_false', dest='use_blur',
                        help='Disable GaussianBlur.')
    parser.set_defaults(use_blur=True)

    # Positional encoding / PB / GCTT-compatible options.
    parser.add_argument('--no_geometric_bias', action='store_false', dest='geometric_bias',
                        help='Disable stratified Fourier initialization in positional encoding.')
    parser.set_defaults(geometric_bias=True)

    parser.add_argument('--no_gctt', action='store_false', dest='use_gctt',
                        help='PB-grid mode: disable oriented-frame SPE and use center-only SPE.')
    parser.set_defaults(use_gctt=True)
    parser.add_argument('--gctt_gauge_jitter_deg', default=0.0, type=float,
                        help='Ignored in pure grid mode. Grid GCTT uses zero gauge for clean ablation.')
    parser.add_argument('--gctt_local_gauge_jitter_deg', default=0.0, type=float,
                        help='Ignored in pure grid mode. Grid GCTT uses zero gauge for clean ablation.')
    parser.add_argument('--gauge_num_frequencies', default=16, type=int,
                        help='Deprecated; kept for CLI/checkpoint compatibility.')
    parser.add_argument('--gauge_scale_init', default=0.02, type=float,
                        help='Deprecated; kept for CLI/checkpoint compatibility.')
    parser.add_argument('--no_gauge_bias', action='store_false', dest='use_gauge_bias',
                        help='Disable oriented-frame relative attention bias term.')
    parser.set_defaults(use_gauge_bias=True)
    parser.add_argument('--gauge_bias_init', default=0.0, type=float,
                        help='Initial beta for oriented-frame relative attention bias.')

    # Masking. Default is MAE-style random masking.
    parser.add_argument('--dynamic_mask_ratio', action='store_true',
                        help='Use a dynamic mask ratio sampled from [0.6, 0.9].')
    parser.set_defaults(dynamic_mask_ratio=False)
    parser.add_argument('--adaptive_masking', action='store_true',
                        help='Enable previous content/latitude-aware masking. Default is plain MAE random masking.')
    parser.add_argument('--no_adaptive_masking', action='store_false', dest='adaptive_masking',
                        help='Disable adaptive masking and use plain MAE random masking.')
    parser.set_defaults(adaptive_masking=False)

    # Training parameters.
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU.')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations.')

    # Model parameters.
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Fixed mask ratio if not using dynamic masking.')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use per-patch normalized pixels as MAE targets.')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--no_angular_bias', action='store_false', dest='use_angular_bias',
                        help='Disable angular-distance attention bias.')
    parser.set_defaults(use_angular_bias=True)
    parser.add_argument('--angular_bias_init_slope', default=1.0, type=float,
                        help='Initial max slope for ALiBi-style angular bias.')

    # Optimizer parameters.
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay.')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Absolute learning rate.')
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR',
                        help='Base LR: absolute_lr = base_lr * total_batch_size / 256.')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR', help='Lower LR bound.')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='Warmup epochs.')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer-wise LR decay.')

    # Checkpoint / paths.
    parser.add_argument('--finetune', default='', type=str, help='Load pretrained weights.')
    parser.add_argument('--resume', default='', type=str, help='Resume checkpoint.')
    parser.add_argument('--data_path', default='./sun360_outdoor/', type=str,
                        help='Path to full ERP panorama images.')
    parser.add_argument('--val_data_path', default=None, type=str,
                        help='Validation dataset path.')
    parser.add_argument('--output_dir', default='./output_pano_grid_recon', type=str,
                        help='Output path.')
    parser.add_argument('--log_dir', default=None, type=str, help='TensorBoard log path.')

    # Dataset / runtime.
    parser.add_argument('--eval_freq', default=10, type=int, help='Evaluation frequency in epochs.')
    parser.add_argument('--device', default='cuda', type=str, help='Device.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory.')
    parser.set_defaults(pin_mem=False)

    # Distributed training.
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)

    return parser


def load_pretrained_weights(model, checkpoint_path):
    print(f'Loading pre-trained ViT weights from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model', checkpoint)

    for k in ['pos_embed', 'head.weight', 'head.bias']:
        if k in checkpoint_model:
            print(f"Removing unused key '{k}' from pre-trained checkpoint.")
            del checkpoint_model[k]

    model_state = model.state_dict()

    # strict=False does not ignore shape mismatches, so remove them explicitly.
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
            print(f'Shape mismatch for {pred_w_key}. Removing decoder prediction head.')
            del checkpoint_model[pred_w_key]
            if pred_b_key in checkpoint_model:
                del checkpoint_model[pred_b_key]

    # Interpolate patch/view embedding kernels when patch size changes.
    pe_keys = [
        k for k in checkpoint_model.keys()
        if ('patch_embed' in k or 'view_embed' in k)
        and 'weight' in k
        and checkpoint_model[k].ndim == 4
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
                print(f'Interpolating {ckpt_key}: {ckpt_weight.shape} -> {model_weight.shape}')
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
    print(f'Restored from pre-trained checkpoint: {msg}')


def _build_scheduler(optimizer, warmup_steps, main_steps, min_lr):
    schedulers = []
    milestones = []
    if warmup_steps > 0:
        schedulers.append(
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)
    schedulers.append(
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, main_steps),
            eta_min=min_lr,
        )
    )
    if len(schedulers) == 1:
        return schedulers[0]
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    v_steps = args.grid_height
    u_steps = 2 * v_steps
    if args.pano_h % v_steps != 0 or args.pano_w % u_steps != 0:
        raise ValueError(
            f'Invalid grid: pano_h={args.pano_h}, pano_w={args.pano_w}, '
            f'grid={v_steps}x{u_steps}. Panorama dimensions must be divisible by the grid.'
        )
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    if patch_h != patch_w:
        raise ValueError(
            f'Patch size must be square for this MAE head, got {patch_h}x{patch_w}. '
            f'For 2:1 ERP, use u_steps=2*grid_height.'
        )
    dynamic_img_size = patch_h

    print(f'Input ERP size: {args.pano_w}x{args.pano_h}')
    print(f'Tokenization: pure ERP grid, no ODI, no tangent-plane')
    print(f'Grid: {v_steps}x{u_steps}, patch size: {dynamic_img_size}x{dynamic_img_size}')
    print(f'Mode: {"GCTT-compatible oriented SPE, zero grid gauge" if args.use_gctt else "PB-grid / center-only SPE"}')
    print(f'Masking: {"adaptive" if args.adaptive_masking else "plain MAE random"}')

    dataset_train = PanoramicDataset(
        root_dir=args.data_path,
        grid_height=args.grid_height,
        img_size=dynamic_img_size,
        pano_h=args.pano_h,
        pano_w=args.pano_w,
        multiscale_sampling=args.multiscale_sampling,
        use_full_pose3d=args.use_full_pose3d,
        use_horizontal_roll=args.use_horizontal_roll,
        use_color_jitter=args.use_color_jitter,
        use_blur=args.use_blur,
        angle_jitter_deg=args.angle_jitter_deg,
        use_gctt=args.use_gctt,
        gctt_gauge_jitter_deg=0.0,
        gctt_local_gauge_jitter_deg=0.0,
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
            use_full_pose3d=False,
            use_horizontal_roll=False,
            use_color_jitter=False,
            use_blur=False,
            use_gctt=args.use_gctt,
            gctt_gauge_jitter_deg=0.0,
            gctt_local_gauge_jitter_deg=0.0,
        )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=False,
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
        adaptive_masking=args.adaptive_masking,
        use_angular_bias=args.use_angular_bias,
        angular_bias_init_slope=args.angular_bias_init_slope,
        use_gctt=args.use_gctt,
        gauge_num_frequencies=args.gauge_num_frequencies,
        gauge_scale_init=args.gauge_scale_init,
        use_gauge_bias=args.use_gauge_bias and args.use_gctt,
        gauge_bias_init=args.gauge_bias_init,
    )

    if args.finetune:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model
    print(f'Model: {args.model}, Parameters (M): '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6:.2f}')

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
    print(f'Peak LR for the schedule: {args.lr:.8f}')

    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    len_loader = len(data_loader_train)
    if len_loader == 0:
        raise RuntimeError('Training dataloader length is 0. Check dataset size, batch_size, and drop_last=True.')
    steps_per_epoch = max(1, len_loader // args.accum_iter)
    if len_loader % args.accum_iter != 0:
        print(f'Warning: len_loader ({len_loader}) is not divisible by accum_iter ({args.accum_iter}).')

    warmup_steps = args.warmup_epochs * steps_per_epoch
    main_steps = max(1, (args.epochs - args.warmup_epochs) * steps_per_epoch)
    print(f'Scheduler: {steps_per_epoch} updates per epoch. '
          f'Warmup for {warmup_steps} updates, then decay for {main_steps} updates.')

    scheduler = _build_scheduler(optimizer, warmup_steps, main_steps, args.min_lr)

    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        resumed_lr = optimizer.param_groups[0]['lr']
        print(f'Resumed training from epoch {args.start_epoch}. Current LR: {resumed_lr:.8f}')

    log_writer = None
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            scheduler,
            device,
            epoch,
            loss_scaler,
            args,
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

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer:
                for k, v in log_stats.items():
                    log_writer.add_scalar(k, v, epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

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
