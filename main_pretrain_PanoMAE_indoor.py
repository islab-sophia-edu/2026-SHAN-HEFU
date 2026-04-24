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

from util.datasets_PanoMAE_legacy import PanoramicDataset
from models_PanoMAE import vit_base_patch16, vit_large_patch16, vit_huge_patch14
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain_PanoMAE import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('Panoramic MAE Finetuning', add_help=False)
    parser.add_argument('--multiscale_sampling', action='store_true', help='Use multiscale sampling data augmentation.')
    parser.set_defaults(multiscale_sampling=False)
    parser.add_argument('--dynamic_mask_ratio', action='store_true', help='Use a dynamic mask ratio [0.6, 0.9] during training.')
    parser.set_defaults(dynamic_mask_ratio=False)
    parser.add_argument('--no_geometric_bias', action='store_false', dest='geometric_bias', help='Disable geometric bias in positional encoding.')
    parser.set_defaults(geometric_bias=True)
    parser.add_argument('--no_adaptive_masking', action='store_false', dest='adaptive_masking', help='Disable adaptive masking strategy.')
    parser.set_defaults(adaptive_masking=True)
    parser.add_argument('--use_horizontal_roll', action='store_true', help='Enable Random Horizontal Roll augmentation (pixel & angle sync).')
    parser.set_defaults(use_horizontal_roll=False)

    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Fixed mask ratio (if not using dynamic).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (L2 regularization)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='absolute learning rate')
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')

    # Finetuning params
    parser.add_argument('--finetune', default='', type=str, help='Finetune from an official MAE pre-trained checkpoint')
    parser.add_argument('--resume', default='', type=str, help='Resume from our own finetuning checkpoint')
    parser.add_argument('--data_path', default='./sun360_outdoor/', type=str, help='Path to the directory of full panoramic images')
    parser.add_argument('--val_data_path', default=None, type=str, help='Validation dataset path')
    parser.add_argument('--grid_height', default=4, type=int, help='Number of patches in vertical direction (e.g., 4 for a 4x8 grid)')
    parser.add_argument('--output_dir', default='./output_pano_recon', type=str, help='Path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, type=str, help='Path where to tensorboard log')
    
    # Dataset parameters
    parser.add_argument('--eval_freq', default=10, type=int, help='Frequency of evaluation in epochs')
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=False)

    #Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)
    
    return parser

def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading pre-trained ViT weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model', checkpoint)
    
    for k in ['pos_embed', 'head.weight', 'head.bias']:
        if k in checkpoint_model:
            print(f"Removing unused key '{k}' from pre-trained checkpoint.")
            del checkpoint_model[k]

    pretrained_pe_weight_key = 'patch_embed.proj.weight'
    model_ve_weight_key = 'view_embed.proj.weight'

    if pretrained_pe_weight_key in checkpoint_model and model_ve_weight_key in model.state_dict():
        pretrained_pe_weight = checkpoint_model[pretrained_pe_weight_key]
        model_ve_weight = model.state_dict()[model_ve_weight_key]
        
        # Check for shape mismatch between pretrained and current model
        if pretrained_pe_weight.shape != model_ve_weight.shape:
            print(f"Patch embedding interpolation: Shape mismatch detected.")
            print(f"Upscaling weights from {pretrained_pe_weight.shape} to {model_ve_weight.shape}")
            
            # Interpolate the 4D weight tensor (B, C, H, W)
            interpolated_weight = F.interpolate(
                pretrained_pe_weight,
                size=model_ve_weight.shape[2:], # Target H, W of the model's embedder
                mode='bicubic',
                align_corners=False
            )
            
            # Update the checkpoint with the new upscaled weight
            checkpoint_model[model_ve_weight_key] = interpolated_weight
            del checkpoint_model[pretrained_pe_weight_key]

            # Also handle the corresponding bias term if it exists
            pretrained_pe_bias_key = 'patch_embed.proj.bias'
            model_ve_bias_key = 'view_embed.proj.bias'
            if pretrained_pe_bias_key in checkpoint_model and model_ve_bias_key in model.state_dict():
                checkpoint_model[model_ve_bias_key] = checkpoint_model[pretrained_pe_bias_key]
                del checkpoint_model[pretrained_pe_bias_key]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Restored from pre-trained checkpoint: {msg}")

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    v_steps = args.grid_height
    dynamic_img_size = 1024 // v_steps

    # Data Setup
    dataset_train = PanoramicDataset(
        root_dir=args.data_path, grid_height=args.grid_height,
        img_size=dynamic_img_size, multiscale_sampling=args.multiscale_sampling,
        use_horizontal_roll=args.use_horizontal_roll,
    )
    dataset_val = None
    if args.val_data_path:
        dataset_val = PanoramicDataset(
            root_dir=args.val_data_path, grid_height=args.grid_height,
            img_size=dynamic_img_size, multiscale_sampling=False,
            use_horizontal_roll=False
        )
    
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False) if dataset_val else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = None
    if dataset_val:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val, batch_size=args.batch_size, # Can use full batch size for val
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )

    # Model Setup
    model = globals()[args.model](
        img_size = dynamic_img_size,
        norm_pix_loss=args.norm_pix_loss,
        geometric_bias=args.geometric_bias,
        adaptive_masking=args.adaptive_masking
    )
    if args.finetune:
        load_pretrained_weights(model, args.finetune)
    model.to(device)
    model_without_ddp = model
    print(f"Model: {args.model}, Parameters (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6:.2f}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # Optimizer, Scheduler, Scaler Setup
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Peak LR for the schedule: {args.lr:.8f}")

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    len_loader = len(data_loader_train)
    steps_per_epoch = len_loader // args.accum_iter
    if len_loader % args.accum_iter != 0:
        print(f"Warning: len_loader ({len_loader}) is not divisible by accum_iter ({args.accum_iter}).")
    
    warmup_steps = args.warmup_epochs * steps_per_epoch
    main_steps = (args.epochs - args.warmup_epochs) * steps_per_epoch
    
    print(f"Scheduler: {steps_per_epoch} updates per epoch. Warmup for {warmup_steps} updates, then decay for {main_steps} updates.")
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=args.min_lr)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Resume Logic
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Successfully restored scheduler state directly from checkpoint.")
        else:
            print("Warning: Scheduler state not found. Manually advancing scheduler's internal clock...")
            steps_to_advance = (args.start_epoch - 1) * steps_per_epoch
            for _ in range(steps_to_advance):
                scheduler.step()
            print(f"Scheduler clock advanced by {steps_to_advance} update steps.")
        resumed_lr = optimizer.param_groups[0]['lr']
        print(f"Resumed training from epoch {args.start_epoch}. Current LR in optimizer is {resumed_lr:.8f}")

    # Tensorboard Logger
    log_writer = None
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # THE NEW TRAINING LOOP
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
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, scheduler=scheduler
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