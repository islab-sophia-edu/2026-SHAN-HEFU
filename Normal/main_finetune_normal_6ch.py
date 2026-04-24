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

# 引用 6通道适配文件
import models_normal_6ch as models_normal_cnn
import engine_normal_6ch as engine_normal
from dataset_stanford_normal_6ch import build_normal_dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Normal Estimation (6CH)', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=4, type=int)
    parser.add_argument('--model', default='vit_huge_patch16', type=str, metavar='MODEL') # 记得改成你的 huge 模型
    
    # 几何参数
    parser.add_argument('--pano_h', default=2048, type=int)
    parser.add_argument('--pano_w', default=4096, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    
    parser.add_argument('--nb_classes', default=3, type=int) 
    
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
        print(f"============== Loading Pretrained Weights (Smart Adapt) ==============")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model_dict = model.state_dict()
    new_dict = {}
    loaded_keys = []
    
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")
        
        # [CRITICAL FIX] 不要跳过 head！
        # 如果我们想利用预训练的输出能力，必须加载它
        if k == 'mask_token': 
            continue
        
        # 如果是 patch_embed，改名为 view_embed (适配您的代码)
        if 'patch_embed.proj' in k: k = k.replace('patch_embed.proj', 'view_embed.proj')

        if k in model_dict:
            target_shape = model_dict[k].shape
            
            # 1. 形状完全匹配 -> 直接加载
            if v.shape == target_shape:
                new_dict[k] = v
                loaded_keys.append(k)
            
            # 2. 通道扩充 (3ch -> 6ch)
            elif 'view_embed.proj.weight' in k and v.shape[1] == 3 and target_shape[1] == 6:
                if misc.is_main_process(): print(f"  [Auto-Expand 3->6] {k}")
                rgb_weight = v
                xyz_weight = torch.zeros_like(v) # 或者用高斯初始化
                v = torch.cat([rgb_weight, xyz_weight], dim=1)
                new_dict[k] = v
                loaded_keys.append(k)

            # 3. Patch Size 插值
            elif 'view_embed.proj.weight' in k and v.shape[1] == target_shape[1]:
                 if misc.is_main_process(): print(f"  [Resize Patch] {k}: {v.shape} -> {target_shape}")
                 v = F.interpolate(v, size=target_shape[2:], mode='bicubic', align_corners=False)
                 new_dict[k] = v
                 loaded_keys.append(k)
            
            # 4. Pos Embed 插值
            elif ('pos_embed' in k or 'angle_pos_embed' in k) and v.shape != target_shape:
                 # 简单跳过，让模型重新学习 PE
                 if misc.is_main_process(): print(f"  [PosEmbed Shape Mismatch - Skip] {k}")
                 continue
            
            # 5. [CRITICAL] Head 形状检查
            # 如果预训练权重的 Head 也是 3 通道 (Normal)，则加载
            elif k.startswith('head') and v.shape != target_shape:
                 if misc.is_main_process(): print(f"  [Head Shape Mismatch - Skip] {k}: {v.shape} != {target_shape}")
                 continue
            
            else:
                if misc.is_main_process(): print(f"  [Mismatch - Skip] {k}: {v.shape} != {target_shape}")
                continue
            
    msg = model.load_state_dict(new_dict, strict=False)
    
    # =========================================================
    # [CRITICAL FIX] Head Surgery (Z-Up Pretrained -> Y-Up Target)
    # 只有在 Head 权重被成功加载后，这一步才有意义！
    # =========================================================
    print(">>> Performing Surgery on Output Head Weights (Corrected Mapping)...")
    with torch.no_grad():
        # 获取最后一层权重
        head_weight = model.head[-1].weight # [3, C, 1, 1]
        head_bias = model.head[-1].bias     # [3]
        
        # 必须 clone，否则原地修改会覆盖数据
        old_weight = head_weight.clone()
        old_bias = head_bias.clone()
        
        # 执行置换 (逻辑确认无误)
        # Target X (0) <--- -1 * Old Y (1)
        head_weight[0] = -1.0 * old_weight[1]
        head_bias[0]   = -1.0 * old_bias[1]

        # Target Y (1) <--- Old Z (2) (垂直轴)
        head_weight[1] = old_weight[2]
        head_bias[1]   = old_bias[2]
        
        # Target Z (2) <--- Old X (0)
        head_weight[2] = old_weight[0]
        head_bias[2]   = old_bias[0]
    
    print(">>> Head Surgery Completed. Coordinates Fully Aligned.")
    
    model.to(device)
    if misc.is_main_process(): 
        print(f"[Success] Loaded {len(loaded_keys)} keys. Missing: {len(msg.missing_keys)}")

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

    dataset_train = build_normal_dataset(is_train=True, args=args)
    dataset_val = build_normal_dataset(is_train=False, args=args)
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0: print('Warning: Dist Eval mismatch')
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

    if misc.is_main_process():
        print(f"Creating PanoNormal model: {args.model} (6-Channel Input)")
    
    # [修复点] 显式传入 in_chans=6
    model = models_normal_cnn.__dict__[args.model](
        img_size=real_patch_size,
        patch_size=real_patch_size, 
        in_chans=6, # 确保模型构建为 6 通道
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
    )
    
    if args.finetune:
        load_pretrained_weights(model, args.finetune, args.device)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    
    criterion = engine_normal.CosineLoss()

    if misc.is_main_process():
        print(f"Start training Normal Estimation for {args.epochs} epochs")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = engine_normal.train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )
        
        test_stats = engine_normal.evaluate(data_loader_val, model, device, criterion, args, epoch)
        
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