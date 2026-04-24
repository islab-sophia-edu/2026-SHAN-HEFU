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

# 引用 3通道适配文件
import models_normal as models_normal_cnn
import engine_normal as engine_normal
from dataset_stanford_normal import build_normal_dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Normal Estimation (3CH)', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=4, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')

    # 几何参数
    parser.add_argument('--pano_h', default=2048, type=int)
    parser.add_argument('--pano_w', default=4096, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    
    parser.add_argument('--data_path', default='./train', type=str)
    parser.add_argument('--val_data_path', default='./test', type=str)
    parser.add_argument('--output_dir', default='./output_normal')
    parser.add_argument('--log_dir', default='./output_normal')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--finetune', default='', help='Path to checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser

def load_pretrained_weights(model, checkpoint_path, device):
    if misc.is_main_process():
        print(f"============== Loading Pretrained Weights (With Axis Swap) ==============")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model_dict = model.state_dict()
    new_dict = {}
    loaded_keys = []
    
    for k, v in state_dict.items():
        # 清理前缀
        k = k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")
        if k == 'mask_token': continue
        if 'patch_embed.proj' in k: k = k.replace('patch_embed.proj', 'view_embed.proj')

        # -----------------------------------------------------------
        # [CRITICAL FIX] 手术式修改位置编码权重 (Z-Up -> Y-Up)
        # -----------------------------------------------------------
        if 'angle_pos_embed.fourier_weights' in k:
            if misc.is_main_process():
                print(f"  [AXIS SWAP] Detected {k}. Permuting columns (0, 2, 1) to match Y-Up...")
            
            # v 的形状是 [Num_Frequencies, 3] -> [x, y, z]
            # 原始 Z-Up: Col 0=X, Col 1=Y(Depth), Col 2=Z(Height)
            # 目标 Y-Up: Col 0=X, Col 1=Y(Height), Col 2=Z(Depth)
            
            # 我们要把原本在 Col 2 (Height) 的权重挪到 Col 1
            # 把原本在 Col 1 (Depth) 的权重挪到 Col 2
            
            # 创建新张量
            v_new = v.clone()
            v_new[:, 1] = v[:, 2] # Old Z (Height) -> New Y
            v_new[:, 2] = - v[:, 1] # Old Y (Depth)  -> New Z
            
            # 注意：可能还需要处理符号问题 (Sign Flip)。
            # 在 Z-Up 中 Y = -cos(lat)sin(lon)
            # 在 Y-Up 中 Z = cos(lat)sin(lon)
            # 它们相差一个负号。我们可以在这里给 Z 轴权重取反，帮助模型更快适应。
            # v_new[:, 2] = -v[:, 1] 
            
            new_dict[k] = v_new
            loaded_keys.append(k)
            continue
        # -----------------------------------------------------------

        if k in model_dict:
            target_shape = model_dict[k].shape
            if v.shape == target_shape:
                new_dict[k] = v
                loaded_keys.append(k)
            elif 'view_embed.proj.weight' in k and v.shape[1] == target_shape[1]:
                 if misc.is_main_process(): 
                     print(f"  [Resize Patch] {k}: {v.shape} -> {target_shape}")
                 v = F.interpolate(v, size=target_shape[2:], mode='bicubic', align_corners=False)
                 new_dict[k] = v
                 loaded_keys.append(k)
            else:
                if misc.is_main_process(): 
                    print(f"  [Skip] {k} mismatch: {v.shape} != {target_shape}")
                continue
            
    msg = model.load_state_dict(new_dict, strict=False)
    
    if misc.is_main_process(): 
        print(f"[Success] Loaded {len(loaded_keys)} keys. Angle PE Axis Swapped.")
        print(f"[Info] Missing keys: {msg.missing_keys}")

def main(args):
    # [Fix 1] 移除开头的硬编码 set_device(0)，交给分布式初始化处理
    # if torch.cuda.is_available(): ... (Deleted)

    # 1. 初始化分布式环境
    misc.init_distributed_mode(args)
    
    # [Fix 1] 根据分配到的 rank 设置 device
    device = torch.device(args.device) # misc 会自动更新 args.device 为 cuda:rank

    # 2. 基础设置
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 3. 计算实际 Batch Size 和 LR
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None: 
        args.lr = args.blr * eff_batch_size / 256
    
    if misc.is_main_process():
        print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
        print(f"actual lr: {args.lr:.2e}")
        print(f"effective batch size: {eff_batch_size}")

    # 4. 计算 Patch Size (适配 PanoMAE 逻辑)
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w) 
    args.input_size = real_patch_size

    # 5. 构建数据集
    dataset_train = build_normal_dataset(is_train=True, args=args)
    dataset_val = build_normal_dataset(is_train=False, args=args)
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0: 
                print('Warning: Dist Eval mismatch')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
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

    # 6. 构建模型
    if misc.is_main_process():
        print(f"Creating PanoNormal model: {args.model} (3-Channel Input)")
    
    model = models_normal_cnn.__dict__[args.model](
        img_size=real_patch_size,
        patch_size=real_patch_size, 
        in_chans=3, 
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
    )
    
    # 7. 加载 Finetune 权重 (如果不是 Resume)
    if args.finetune and not args.resume:
        load_pretrained_weights(model, args.finetune, args.device)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # 8. 优化器 & Loss Scaler
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = engine_normal.CosineLoss()

    # [Fix 2] 添加 Resume 逻辑 (至关重要)
    if args.resume:
        if misc.is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        
        # 使用 misc 中的 load_model 或手动加载
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if misc.is_main_process():
                print(f"Resume successful. Start epoch: {args.start_epoch}")
        else:
            if misc.is_main_process():
                print("Warning: Resume checkpoint lacks optimizer/epoch info. Starting fresh.")

    # 9. 训练循环
    if misc.is_main_process():
        print(f"Start training Normal Estimation for {args.epochs} epochs")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = engine_normal.train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )
        
        # Save frequency: Every 20 epochs OR Last epoch
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
             misc.save_model(
                 args=args, model=model, model_without_ddp=model_without_ddp, 
                 optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
             )

        # Evaluate
        test_stats = engine_normal.evaluate(data_loader_val, model, device, criterion, args, epoch)
        
        # Logging
        if misc.is_main_process():
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