import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.lr_decay_rotation as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from PanoMAE_classification_dataset_rotation_legacy import build_classification_dataset
from models_PanoMAE_classification_rotation_easy import vit_huge_patch14
from engine_PanoMAE_classification_rotation_legacy import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('PanoMAE Classification FT', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='vit_huge_patch14', type=str)
    parser.add_argument('--nb_classes', default=32, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--global_pool', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--grid_height', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true', default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    return parser

def build_optimizer_with_head_encoder_split(model, args):
    head_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith('head'): head_params.append(param)
        else: encoder_params.append(param)
        
    param_groups = lrd.param_groups_lrd(
        model, args.weight_decay, 
        no_weight_decay_list=model.no_weight_decay(), 
        layer_decay=args.layer_decay
    )
    
    # 【修复 Bug 3】计算 Head 相对于 Encoder 的学习率倍率
    lr_ratio = args.head_lr / args.encoder_lr  # 例如: 1e-3 / 1e-5 = 100.0
    
    for group in param_groups:
        is_head = any([id(p) in [id(hp) for hp in head_params] for p in group['params']])
        if is_head:
            # 使用 lr_scale 让 lr_sched 自动计算出 1e-3
            group['lr_scale'] = lr_ratio
            
    # 【重点】将 base_lr 设为 encoder_lr，这样 lr_sched 乘出来的结果才对
    return torch.optim.AdamW(param_groups, lr=args.encoder_lr)

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    
    if args.lr is None:
        args.lr = args.encoder_lr 

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_classification_dataset(is_train=True, args=args)
    dataset_val = build_classification_dataset(is_train=False, args=args, class_to_idx=dataset_train.class_to_idx)
    args.nb_classes = len(dataset_train.class_to_idx)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    model = vit_huge_patch14(img_size=512 // args.grid_height, num_classes=args.nb_classes, drop_path_rate=0.1, global_pool=args.global_pool)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = build_optimizer_with_head_encoder_split(model_without_ddp, args)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    history_loss = []
    
    # [LOG FIX] 初始化日志写入
    if misc.is_main_process():
        log_file = os.path.join(args.output_dir, "log.txt")
        if args.start_epoch == 0:
            with open(log_file, "w") as f: f.write("") 

    # 统计训练时间
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: data_loader_train.sampler.set_epoch(epoch)
        dataset_train.set_aug_stage('yaw_only' if epoch < 10 else 'mid_pose' if epoch < 20 else 'full')

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, max_norm=args.clip_grad, args=args)
        history_loss.append(train_stats['loss'])

        test_stats = evaluate(data_loader_val, model, device)

        # 早期失败检测
        need_expand = torch.tensor(0, device=device)
    
        if epoch == 2 and misc.is_main_process():
            acc1_val = train_stats.get('acc1', 0)
            if acc1_val < (100.0 / args.nb_classes + 5.0) and abs(history_loss[0] - history_loss[-1]) < 0.05:
                need_expand = torch.tensor(1, device=device)

        # Sync diagnostic decision across all DDP ranks
        if args.distributed:
            torch.distributed.broadcast(need_expand, src=0)

        if need_expand.item() == 1:
            if misc.is_main_process():
                print("\n[ALERT] Expanding Head to 4D...\n")
            model_without_ddp.expand_head_to_4d()
            optimizer = build_optimizer_with_head_encoder_split(model_without_ddp, args)

        # [LOGGING] 写入 log.txt
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if misc.is_main_process():
            with open(log_file, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 每 20 个 epoch 保存一次权重，或是最后一个 epoch
            current_epoch_num = epoch + 1
            if current_epoch_num % 20 == 0 or current_epoch_num == args.epochs:
                save_path = os.path.join(args.output_dir, f'checkpoint-epoch{current_epoch_num}.pth')
                misc.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, save_path)
                print(f"Saved checkpoint: {save_path}")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time}')

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