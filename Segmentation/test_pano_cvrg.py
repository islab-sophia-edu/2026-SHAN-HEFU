import argparse
import os
import time
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image

import util.misc as misc
import util.odi_processing as odi_helper
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# 引入您现有的模块
import Segmentation.models_segmentation_2dcnn as models_segmentation
from datasets_segmentation import build_segmentation_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Segmentation Testing', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Test batch size')
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--pano_h', default=2048, type=int)
    parser.add_argument('--pano_w', default=4096, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--nb_classes', default=8, type=int)
    parser.add_argument('--data_path', default='./train', type=str)
    parser.add_argument('--val_data_path', default='./test', type=str, help='Path to test/val dataset')
    parser.add_argument('--resume', default='', required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', default='./test_results', help='Path to save visualization')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--drop_path', type=float, default=0.0)
    
    # 兼容性参数
    parser.add_argument('--input_size', default=None, type=int) 
    return parser

# --- 辅助函数：颜色板 ---
def get_color_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def colorize_mask(mask_tensor, palette):
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask_np).convert('P')
    img.putpalette(palette)
    return np.array(img.convert('RGB'))

def stitch_patches(patches_list, args):
    """
    patches_list: List of np.array (H, W, 3) or (H, W)
    """
    first_patch = patches_list[0]
    # 如果是 (H, W) 的单通道 Label，先伪装成 RGB
    if first_patch.ndim == 2:
        patches_list = [np.stack([p, p, p], axis=-1) for p in patches_list]
    
    # 此时 patches_list 应该是 List[H, W, 3]
    pano = odi_helper.embed_patches_to_pano(
        patches_np_list=patches_list,
        pano_size=(args.pano_h, args.pano_w),
        h_num=args.grid_height * 2,
        v_num=args.grid_height,
        h_fov_deg=360.0 / (args.grid_height * 2),
        v_fov_deg=180.0 / args.grid_height
    )
    return pano

def compute_metrics(pred, target, num_classes):
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    target_counts = torch.zeros(num_classes, device=pred.device)

    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds & target_inds).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
        target_counts[cls] = target_inds.sum()
        
    return intersection, union, target_counts

@torch.no_grad()
def run_test(data_loader, model, device, criterion, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    total_inter = torch.zeros(args.nb_classes, device=device)
    total_union = torch.zeros(args.nb_classes, device=device)
    total_target = torch.zeros(args.nb_classes, device=device)
    
    palette = get_color_palette(args.nb_classes)
    
    # 用于反归一化 RGB
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    print(f"Start processing {len(data_loader)} batches...")
    
    # 遍历每一个 Batch (建议 batch_size=1 以便保存图片)
    for batch_idx, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Forward
        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)
            B, N, C = logits.shape[:3]
            
            # 上采样
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, logits.shape[-2], logits.shape[-1]),
                size=(patch_h, patch_w),
                mode='bilinear', align_corners=False
            ) # (B*N, C, H, W)
            
            targets_flat = targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]).float()
            targets_resized = F.interpolate(targets_flat, size=(patch_h, patch_w), mode='nearest').long().squeeze(1)

            loss = criterion(logits_upsampled, targets_resized)
        
        metric_logger.update(loss=loss.item())
        
        # 2. Metrics
        pred_labels = logits_upsampled.argmax(dim=1)
        inter, union, target_cnt = compute_metrics(pred_labels, targets_resized, args.nb_classes)
        total_inter += inter
        total_union += union
        total_target += target_cnt
        
        # 3. Visualization (保存每一张图片)
        # 还原 RGB (B, N, 3, H, W)
        raw_rgb = views * std + mean
        raw_rgb = torch.clamp(raw_rgb, 0, 1)
        
        # Reshape for access
        pred_labels_vis = pred_labels.view(B, N, patch_h, patch_w)
        targets_vis = targets_resized.view(B, N, patch_h, patch_w)
        
        for b in range(B):
            rgb_patches = []
            gt_patches = []
            pred_patches = []
            
            for i in range(N):
                # RGB
                p_rgb = raw_rgb[b, i].permute(1, 2, 0).cpu().float().numpy()
                rgb_patches.append(p_rgb)
                
                # GT
                p_gt = colorize_mask(targets_vis[b, i], palette)
                gt_patches.append(p_gt.astype(np.float32) / 255.0)
                
                # Pred
                p_pred = colorize_mask(pred_labels_vis[b, i], palette)
                pred_patches.append(p_pred.astype(np.float32) / 255.0)
            
            # 拼接全景图
            pano_rgb = stitch_patches(rgb_patches, args)
            pano_gt = stitch_patches(gt_patches, args)
            pano_pred = stitch_patches(pred_patches, args)
            
            # 组合保存
            img_rgb = Image.fromarray((pano_rgb * 255).astype(np.uint8))
            img_gt = Image.fromarray((pano_gt * 255).astype(np.uint8))
            img_pred = Image.fromarray((pano_pred * 255).astype(np.uint8))
            
            total_w = img_rgb.width
            total_h = img_rgb.height * 3
            combined = Image.new('RGB', (total_w, total_h))
            combined.paste(img_rgb, (0, 0))
            combined.paste(img_gt, (0, img_rgb.height))
            combined.paste(img_pred, (0, img_rgb.height * 2))
            
            save_name = f'test_batch{batch_idx}_sample{b}.png'
            save_path = os.path.join(args.output_dir, save_name)
            combined.save(save_path)

    # 4. Final Metrics
    iou_per_class = total_inter / (total_union + 1e-6)
    miou = iou_per_class.mean().item()
    acc_per_class = total_inter / (total_target + 1e-6)
    macc = acc_per_class.mean().item()
    
    print(f"Test Finished. mIoU: {miou:.4f}, mAcc: {macc:.4f}")
    return miou, macc

def main(args):
    # Setup Device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Dataset Setup (Ensure Input Size logic matches main_finetune)
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w) 
    args.input_size = real_patch_size
    print(f"Calculated Patch Size: {real_patch_size}")

    dataset_val = build_segmentation_dataset(is_train=False, args=args)
    print(f"Test Dataset size: {len(dataset_val)}")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem, 
        drop_last=False
    )

    # 2. Model Setup
    print(f"Creating model: {args.model}")
    model = models_segmentation.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=real_patch_size,
        patch_size=real_patch_size, 
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        drop_path_rate=args.drop_path,
    )
    
    # 3. Load Checkpoint
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(f"Loaded checkpoint from {args.resume}")
    print(f"Load msg: {msg}")
    
    model.to(device)

    # 4. Criterion (CrossEntropy + Lovasz)
    # 保持和训练一致的 Loss 设置，虽然测试时只看指标，但计算 Loss 可以作为参考
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 5. Run Test
    run_test(data_loader_val, model, device, criterion, args)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)