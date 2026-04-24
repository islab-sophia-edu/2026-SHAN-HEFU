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
from torch.utils.tensorboard import SummaryWriter

# [修改] 引用新的 6通道适配文件
import models_segmentation_6ch as models_segmentation
import engine_segmentation_6ch as engine_segmentation

from dataset_stanford_6ch import build_segmentation_dataset 

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Segmentation Finetuning (Stanford 6CH)', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=4, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')
    
    # 几何参数
    parser.add_argument('--pano_h', default=2048, type=int)
    parser.add_argument('--pano_w', default=4096, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--input_size', default=None, type=int, help='Patch size (auto-calculated if None)')
    
    parser.add_argument('--nb_classes', default=13, type=int) # Stanford 2D3DS 通常是 13 类
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')

    parser.add_argument('--data_path', default='./train', type=str)
    parser.add_argument('--val_data_path', default='./test', type=str)
    parser.add_argument('--output_dir', default='./output_seg_stanford')
    parser.add_argument('--log_dir', default='./output_seg_stanford')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--finetune', default='', help='Path to pretrained checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser

def load_pretrained_weights(model, checkpoint_path):
    if misc.is_main_process():
        print(f"============== Loading Pretrained Weights ==============")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model_dict = model.state_dict()
    new_dict = {}
    loaded_keys = []
    
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        # 跳过不匹配的层 (Decoder, Head)
        if 'decoder' in k or 'mask_token' in k or 'head' in k: continue
        
        # 映射 key 名称
        if 'patch_embed.proj' in k: k = k.replace('patch_embed.proj', 'view_embed.proj')

        if k in model_dict:
            target_shape = model_dict[k].shape
            if v.shape != target_shape:
                # [核心修复] 处理 3通道 -> 6通道 的权重迁移
                if 'view_embed.proj.weight' in k:
                    if misc.is_main_process(): 
                        print(f"  [Channel Expand] {k}: {v.shape} -> {target_shape}")
                    
                    # 1. 扩充通道: RGB (copy) + XYZ (zero init)
                    # v: (768, 3, 32, 32) -> target: (768, 6, 32, 32)
                    rgb_weight = v
                    xyz_weight = torch.zeros_like(v) # 初始化为 0
                    
                    # 拼接到 6 通道
                    v_expanded = torch.cat([rgb_weight, xyz_weight], dim=1) 
                    
                    # 2. 插值 Patch Size (如果 Patch Size 也不一样)
                    if v_expanded.shape[2:] != target_shape[2:]:
                        if misc.is_main_process():
                            print(f"  [Patch Resize] {k}: {v_expanded.shape} -> {target_shape}")
                        v_expanded = F.interpolate(
                            v_expanded, size=target_shape[2:], 
                            mode='bicubic', align_corners=False
                        )
                    
                    v = v_expanded

                elif 'pos_embed' in k or 'angle_pos_embed' in k:
                     if misc.is_main_process(): print(f"  [Skip Shape Mismatch] {k}: {v.shape} != {target_shape}")
                     continue
                else:
                    if misc.is_main_process(): print(f"  [Mismatch] Key: {k} | Ckpt: {v.shape} != Model: {target_shape}")
                    continue
            
            new_dict[k] = v
            loaded_keys.append(k)
            
    msg = model.load_state_dict(new_dict, strict=False)
    if misc.is_main_process(): 
        print(f"[Success] Loaded {len(loaded_keys)} keys. Missing: {msg.missing_keys[:5]}...")

def main(args):
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0) 

    misc.init_distributed_mode(args)
    if args.device == 'cuda': args.device = 'cuda:0'
    device = torch.device(args.device)
    
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # LR Calc
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None: 
        args.lr = args.blr * eff_batch_size / 256
    
    if misc.is_main_process():
        print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
        print(f"actual lr: {args.lr:.2e}")
        print(f"effective batch size: {eff_batch_size}")

    # Dataset Setup (Ensure Input Size is Correct)
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w) 
    
    # 强制更新 args.input_size 供 Dataset 使用
    args.input_size = real_patch_size

    # 调用 dataset_stanford 的 build 函数
    # 注意：您需要确保 dataset_stanford.py 已经被修改为返回 6通道数据
    dataset_train = build_segmentation_dataset(is_train=True, args=args)
    dataset_val = build_segmentation_dataset(is_train=False, args=args)
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0: print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem, 
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem, 
        drop_last=False
    )

    # Model Setup
    if misc.is_main_process():
        print(f"Creating model: {args.model} (6-Channel Input)")
    
    model = models_segmentation.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=real_patch_size,
        patch_size=real_patch_size, 
        in_chans=6, # [关键] 6通道输入
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
    )
    
    if args.finetune:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Optimizer
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)

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
            if epoch % 10 == 0 or epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
                                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if misc.is_main_process():
        print('Training time {}'.format(total_time_str))

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