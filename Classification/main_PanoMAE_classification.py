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

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchvision import transforms

# Custom Modules
#from PanoMAE_classification_dataset import build_classification_dataset
from PanoMAE_classification_dataset import build_classification_dataset
from models_PanoMAE_classification import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from engine_PanoMAE_classification import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('PanoMAE Classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')
    # Note: nb_classes will be auto-updated by dataset
    parser.add_argument('--nb_classes', default=32, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Augmentation
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--recount', type=int, default=1)

    # IO & Sys
    parser.add_argument('--finetune', default='')
    parser.add_argument('--data_path', default='./sun360/train', type=str)
    parser.add_argument('--val_data_path', default='./sun360/val', type=str)
    parser.add_argument('--grid_height', default=4, type=int)
    parser.add_argument('--output_dir', default='./output_dir_cls')
    parser.add_argument('--log_dir', default='./output_dir_cls')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    
    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser

def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    
    # 1. 智能去前缀 & 过滤
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 DDP/Model 前缀
        k = k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")
        
        # 移除分类头和Mask Token (微调不需要)
        if k.startswith('head') or k.startswith('decoder_') or k == 'mask_token':
            continue
            
        new_state_dict[k] = v
        
    # 2. 形状检查与插值 (Handle Resolution Mismatch)
    model_dict = model.state_dict()
    for k, v in new_state_dict.items():
        if k in model_dict and v.shape != model_dict[k].shape:
            # 处理 PatchEmbed (view_embed) 尺寸不匹配
            if 'view_embed' in k and 'weight' in k:
                print(f"Interpolating {k} from {v.shape} to {model_dict[k].shape}")
                # v: [O, I, H, W] -> interpolate last 2 dims
                v = torch.nn.functional.interpolate(v, size=model_dict[k].shape[2:], mode='bicubic', align_corners=False)
                new_state_dict[k] = v
            else:
                print(f"[WARNING] Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}. Skipping.")
                continue

    # 3. 加载并严格验证
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    # 4. 致命错误检查：核心权重是否加载？
    # 你的模型叫 'view_embed'，预训练也叫 'view_embed'，必须检查它！
    missing_keys = msg.missing_keys
    critical_missing = [k for k in missing_keys if 'view_embed' in k or 'blocks.0.' in k]
    
    if len(critical_missing) > 0:
        print("\n" + "="*40)
        print("[CRITICAL ERROR] Pre-trained weights NOT loaded correctly!")
        print(f"Missing Critical Keys Sample: {critical_missing[:5]}")
        print("="*40 + "\n")
        # 抛出异常，防止这一步只是打印而你没看到，导致白跑几小时
        raise RuntimeError("Critical backbone weights are missing. Check checkpoint compatibility.")
    
    print(f"[SUCCESS] Pre-trained weights loaded successfully. Missing keys: {len(missing_keys)} (Expected: head, etc.)")

def save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, max_accuracy, tag):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f'checkpoint-{tag}.pth'
    
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'epoch': epoch,
        'max_accuracy': max_accuracy, # Back to standard Acc1
        'args': args,
    }
    misc.save_on_master(to_save, checkpoint_path)

def main(args):
    misc.init_distributed_mode(args)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 1. Dataset Construction (Logic to handle 31 vs 25)
    print("Building Training Dataset...")
    dataset_train = build_classification_dataset(is_train=True, args=args)
    
    # Get the canonical class mapping from Train
    train_mapping = dataset_train.class_to_idx
    print(f"Training Class Mapping: {len(train_mapping)} classes (others excluded).")
    
    print("Building Validation Dataset...")
    # Pass mapping to Validation to enforce alignment
    dataset_val = build_classification_dataset(is_train=False, args=args, class_to_idx=train_mapping)

    # Auto-update nb_classes
    if args.nb_classes != len(train_mapping):
        print(f"Info: Updating nb_classes from {args.nb_classes} to {len(train_mapping)}")
        args.nb_classes = len(train_mapping)

    # 2. Sampler (Standard Distributed or Random)
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 3. Loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, drop_last=False
    )

    # 4. Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 5. Model
    v_steps = args.grid_height
    dynamic_img_size = 512 // v_steps

    model = globals()[args.model](
        img_size=dynamic_img_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        load_pretrained_weights(model, args.finetune)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 6. Optimizer
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    print(f"Base LR: {args.lr:.2e}")

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    
    print("Applying Layer-wise Learning Rate Decay...")
    for group in optimizer.param_groups:
        if 'lr_scale' in group:
            group['lr'] = group['lr'] * group['lr_scale']
            
    loss_scaler = NativeScaler()

    # 7. Loss (Standard)
    if mixup_fn is not None:
        # Standard SoftTarget CE for Mixup
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Criterion: {criterion}")

    # 8. Resume
    max_accuracy = 0.0 
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        
        if 'epoch' in checkpoint and isinstance(checkpoint['epoch'], int):
            args.start_epoch = checkpoint['epoch'] + 1
        
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy: {test_stats['acc1']:.2f}%")
        exit(0)

    # 9. Train
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    log_writer = None
    if misc.is_main_process() and args.log_dir:
        try:
            log_writer = SummaryWriter(log_dir=args.log_dir)
        except:
            pass

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=args.clip_grad,
            mixup_fn=mixup_fn,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy: {test_stats['acc1']:.2f}%")
        
        if args.output_dir:
            # Save Last
            save_checkpoint(
                args, epoch, model, model_without_ddp, optimizer, loss_scaler, max_accuracy, tag="last"
            )

            # Save Best Acc1
            if test_stats["acc1"] > max_accuracy:
                max_accuracy = test_stats["acc1"]
                save_checkpoint(
                    args, epoch, model, model_without_ddp, optimizer, loss_scaler, max_accuracy, tag="best"
                )
                print(f'>> New Best Acc1: {max_accuracy:.2f}% (Saved)')

            # Periodic
            if epoch % 20 == 0 or epoch + 1 == args.epochs:
                tag = f"{epoch:04d}"
                save_checkpoint(
                    args, epoch, model, model_without_ddp, optimizer, loss_scaler, max_accuracy, tag=tag
                )

        if log_writer:
            try:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            except:
                pass

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            try:
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            except:
                pass

    print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))

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