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
import glob

import util.misc as misc
import util.odi_processing as odi_helper

# 尝试导入模型定义，优先使用带 CNN 的版本
try:
    import Segmentation.models_segmentation_2dcnn as models_segmentation
    print("Imported models_segmentation_cnn successfully.")
except ImportError:
    import models_segmentation
    print("Imported models_segmentation successfully.")

from torchvision import transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Stanford2D3D Pano Segmentation Testing', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Test batch size')
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')
    
    # 您的启动命令参数
    parser.add_argument('--pano_h', default=1024, type=int)
    parser.add_argument('--pano_w', default=2048, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--nb_classes', default=14, type=int)
    
    parser.add_argument('--val_data_path', default='./test', type=str, help='Path to test/val dataset root (containing rgb and anno folders)')
    parser.add_argument('--resume', default='', required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', default='./test_results_stanford', help='Path to save visualization')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--drop_path', type=float, default=0.0)
    
    parser.add_argument('--input_size', default=None, type=int) 
    return parser

# --- Stanford2D3D Direct Dataset (针对您的路径结构定制) ---
class Stanford2D3DTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, grid_height=16, args=None):
        self.root_dir = root_dir
        self.args = args
        
        # 直接在 root_dir 下找 rgb 和 anno
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.mask_dir = os.path.join(root_dir, 'anno') # 别找 mask，找 anno
        
        print(f"Loading Test Data from Direct Path:")
        print(f"  - RGB:  {self.rgb_dir}")
        print(f"  - ANNO: {self.mask_dir}")
        
        if not os.path.exists(self.rgb_dir) or not os.path.exists(self.mask_dir):
            raise ValueError(f"Error: rgb or anno folder not found in {root_dir}")

        self.filenames = []
        rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, '*.png')))
        
        for rgb_path in rgb_files:
            file_name = os.path.basename(rgb_path)
            # 命名规则: camera_xxx_rgb.png -> camera_xxx_anno.png
            mask_name = file_name.replace('_rgb.png', '_anno.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.filenames.append({'rgb': rgb_path, 'mask': mask_path})
            else:
                # 尝试容错：有些文件可能叫 _semantic.png
                mask_name_alt = file_name.replace('_rgb.png', '_semantic.png')
                mask_path_alt = os.path.join(self.mask_dir, mask_name_alt)
                if os.path.exists(mask_path_alt):
                    self.filenames.append({'rgb': rgb_path, 'mask': mask_path_alt})

        print(f"Found {len(self.filenames)} valid pairs.")

        # Grid Setup
        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        
        v_centers = torch.linspace(90 - self.v_fov/2, -90 + self.v_fov/2, self.v_steps)
        u_centers = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1) 

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.filenames[idx]
        try:
            img_pil = Image.open(data['rgb']).convert('RGB')
            mask_pil = Image.open(data['mask']).convert('RGB') # 先读 RGB

            # Resize
            if self.args and hasattr(self.args, 'pano_h'):
                target_h, target_w = self.args.pano_h, self.args.pano_w
                if img_pil.size != (target_w, target_h):
                    img_pil = img_pil.resize((target_w, target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)

            img_np = np.array(img_pil)
            mask_raw = np.array(mask_pil)

            # Mask 处理: 直接取第一个通道 (因为 anno 是索引图 R=G=B)
            mask_np = mask_raw[:, :, 0].astype(np.int64)

            # Patch Extraction
            raw_h = img_np.shape[0] // self.v_steps
            raw_w = img_np.shape[1] // self.u_steps
            target_size = (int(raw_w), int(raw_h)) if raw_h != raw_w else int(raw_h)

            rgb_patches_list = odi_helper.extract_patches_from_pano(
                pano_np=img_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            mask_patches_list = odi_helper.extract_patches_from_pano(
                pano_np=mask_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=target_size)
            
            rgb_tensors = []
            mask_tensors = []
            
            for i in range(len(rgb_patches_list)):
                p_rgb = Image.fromarray(rgb_patches_list[i])
                p_rgb_tensor = self.normalize(transforms.ToTensor()(p_rgb))
                rgb_tensors.append(p_rgb_tensor)

                p_mask_np = mask_patches_list[i]
                mask_tensors.append(torch.from_numpy(p_mask_np.astype(np.int64)))

            patches_img = torch.stack(rgb_tensors)   
            patches_mask = torch.stack(mask_tensors)

        except Exception as e:
            print(f"Error loading {data['rgb']}: {e}")
            # Dummy return
            h_fb = self.args.pano_h // self.v_steps
            w_fb = self.args.pano_w // self.u_steps
            return (torch.zeros(self.u_steps * self.v_steps, 3, h_fb, w_fb),
                    self.angle_centers,
                    torch.zeros(self.u_steps * self.v_steps, h_fb, w_fb).long())

        return patches_img, self.angle_centers, patches_mask

# --- Helpers ---
def get_stanford_palette():
    # 13 类标准颜色 + 黑色
    stanford_colors = [
        128, 128, 128, # 0: beam
        128, 0, 0,     # 1: board
        128, 64, 128,  # 2: bookcase
        0, 0, 192,     # 3: ceiling
        64, 64, 128,   # 4: chair
        128, 128, 0,   # 5: clutter
        192, 192, 128, # 6: column
        128, 0, 128,   # 7: door
        128, 64, 0,    # 8: floor
        0, 192, 192,   # 9: table
        0, 128, 0,     # 10: wall
        0, 0, 128,     # 11: window
        0, 128, 128,   # 12: sofa
        0, 0, 0        # 13: <unk>
    ]
    stanford_colors += [0] * (768 - len(stanford_colors))
    return stanford_colors

def colorize_mask(mask_tensor, palette):
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask_np).convert('P')
    img.putpalette(palette)
    return np.array(img.convert('RGB'))

def stitch_patches(patches_list, args):
    first_patch = patches_list[0]
    if first_patch.ndim == 2:
        patches_list = [np.stack([p, p, p], axis=-1) for p in patches_list]
    
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
    # Ignore index 13 (unk) and 255
    ignore_indices = [13, 255]
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    for ig in ignore_indices:
        valid_mask &= (target != ig)
    
    pred = pred[valid_mask]
    target = target[valid_mask]

    valid_classes = num_classes - 1 # 排除第13类
    intersection = torch.zeros(valid_classes, device=pred.device)
    union = torch.zeros(valid_classes, device=pred.device)
    target_counts = torch.zeros(valid_classes, device=pred.device)
    
    for cls in range(valid_classes):
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
    
    valid_classes = args.nb_classes - 1
    total_inter = torch.zeros(valid_classes, device=device)
    total_union = torch.zeros(valid_classes, device=device)
    total_target = torch.zeros(valid_classes, device=device)
    
    palette = get_stanford_palette()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    print(f"Processing...")
    
    for batch_idx, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 5, header)):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
            logits = model(views, angles)
            B, N, C = logits.shape[:3]
            
            patch_h = args.pano_h // args.grid_height
            patch_w = args.pano_w // (2 * args.grid_height)
            
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, logits.shape[-2], logits.shape[-1]),
                size=(patch_h, patch_w),
                mode='bilinear', align_corners=False
            )
            
            targets_flat = targets.view(B*N, 1, targets.shape[-2], targets.shape[-1]).float()
            targets_resized = F.interpolate(targets_flat, size=(patch_h, patch_w), mode='nearest').long().squeeze(1)
            
            # Loss for reference
            loss = criterion(logits_upsampled, targets_resized)

        metric_logger.update(loss=loss.item())
        
        pred_labels = logits_upsampled.argmax(dim=1)
        inter, union, target_cnt = compute_metrics(pred_labels, targets_resized, args.nb_classes)
        total_inter += inter
        total_union += union
        total_target += target_cnt
        
        # --- Save Visualization ---
        # 始终保存图片，因为这是 Test 脚本
        raw_rgb = views * std + mean
        raw_rgb = torch.clamp(raw_rgb, 0, 1)
        
        pred_labels_vis = pred_labels.view(B, N, patch_h, patch_w)
        targets_vis = targets_resized.view(B, N, patch_h, patch_w)
        
        for b in range(B):
            rgb_patches = []
            gt_patches = []
            pred_patches = []
            
            for i in range(N):
                p_rgb = raw_rgb[b, i].permute(1, 2, 0).cpu().float().numpy()
                rgb_patches.append(p_rgb)
                
                p_gt = colorize_mask(targets_vis[b, i], palette)
                gt_patches.append(p_gt.astype(np.float32) / 255.0)
                
                p_pred = colorize_mask(pred_labels_vis[b, i], palette)
                pred_patches.append(p_pred.astype(np.float32) / 255.0)
            
            pano_rgb = stitch_patches(rgb_patches, args)
            pano_gt = stitch_patches(gt_patches, args)
            pano_pred = stitch_patches(pred_patches, args)
            
            img_rgb = Image.fromarray((pano_rgb * 255).astype(np.uint8))
            img_gt = Image.fromarray((pano_gt * 255).astype(np.uint8))
            img_pred = Image.fromarray((pano_pred * 255).astype(np.uint8))
            
            # 拼接: 上RGB 中GT 下Pred
            total_w = img_rgb.width
            total_h = img_rgb.height * 3
            combined = Image.new('RGB', (total_w, total_h))
            combined.paste(img_rgb, (0, 0))
            combined.paste(img_gt, (0, img_rgb.height))
            combined.paste(img_pred, (0, img_rgb.height * 2))
            
            save_name = f'test_batch{batch_idx}_{b}.png'
            save_path = os.path.join(args.output_dir, save_name)
            combined.save(save_path)
            # print(f"Saved {save_path}")

    iou_per_class = total_inter / (total_union + 1e-6)
    miou = iou_per_class.mean().item()
    acc_per_class = total_inter / (total_target + 1e-6)
    macc = acc_per_class.mean().item()
    
    print("="*40)
    print(f"Per-Class IoU (0-12): {iou_per_class.cpu().numpy()}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Mean Acc: {macc:.4f}")
    print("="*40)

def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Dataset Setup
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w) 
    args.input_size = real_patch_size
    print(f"Resolution: {args.pano_h}x{args.pano_w} | Patch: {real_patch_size}")

    # 使用新的 Test Dataset
    dataset_val = Stanford2D3DTestDataset(
        root_dir=args.val_data_path,
        grid_height=args.grid_height,
        args=args
    )
    print(f"Test Images: {len(dataset_val)}")

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
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 去除 DDP 前缀 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 尝试加载
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights from {args.resume}")
    print(f"Missing keys (expect none or only head related if mismatch): {msg.missing_keys}")
    
    model.to(device)

    # 4. Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 5. Run
    run_test(data_loader_val, model, device, criterion, args)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)