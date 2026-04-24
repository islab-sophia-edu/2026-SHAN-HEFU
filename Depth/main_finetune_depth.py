import argparse
import os
import time
import json
import datetime
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# [修改] 导入 Depth 相关的模块
import models_depth as models_depth_cnn
import engine_depth as engine_depth
from dataset_stanford_depth import build_depth_dataset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_decay as lrd

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Depth Estimation', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=4, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--pano_h', default=2048, type=int)
    parser.add_argument('--pano_w', default=4096, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    
    # [移除] nb_classes (Depth 是回归)
    
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
    parser.add_argument('--output_dir', default='./output_depth')
    parser.add_argument('--log_dir', default='./output_depth')
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

    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w) 
    args.input_size = real_patch_size

    dataset_train = build_depth_dataset(is_train=True, args=args)
    dataset_val = build_depth_dataset(is_train=False, args=args)
    
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

    if misc.is_main_process():
        print(f"Creating PanoDepth model: {args.model}")
    
    model = models_depth_cnn.__dict__[args.model](
        img_size=real_patch_size,
        patch_size=real_patch_size, 
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
    )
    
    if args.finetune:
        if misc.is_main_process():
            print(f"Load pre-trained checkpoint from: {args.finetune}")
        # [修正] 加上 weights_only=False
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        # 移除 head 权重
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        
        # 简单处理 interpolate (可选，如果遇到 patch_embed 维度不匹配可以加上)
        model_dict = model.state_dict()
        for k, v in state_dict.items():
            if k in model_dict and v.shape != model_dict[k].shape:
                if 'patch_embed.proj.weight' in k:
                    if misc.is_main_process():
                        print(f"Interpolating {k} from {v.shape} to {model_dict[k].shape}")
                    v = F.interpolate(v, size=model_dict[k].shape[2:], mode='bicubic', align_corners=False)
                    state_dict[k] = v

        msg = model.load_state_dict(state_dict, strict=False)
        if misc.is_main_process():
            print(f"Loaded keys. Missing: {len(msg.missing_keys)}")

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
    
    criterion = engine_depth.DepthLoss(w_silog=1.0, w_berhu=0.5, w_grad=0.5).to(device)

    if misc.is_main_process():
        print(f"Start training Depth Estimation for {args.epochs} epochs")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = engine_depth.train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )
        
        test_stats = engine_depth.evaluate(data_loader_val, model, device, criterion, args, epoch)
        
        if misc.is_main_process():
            if epoch % 20 == 0 or epoch + 1 == args.epochs:
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