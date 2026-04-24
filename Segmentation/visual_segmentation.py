import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image

import util.misc as misc
import util.odi_processing_gpu as odi_helper
import models_segmentation_2dcnn as models_segmentation
from datasets_segmentation_8class import build_segmentation_dataset

# ==========================================
# 目标候选图片列表 (仅在此列表中寻找 Top 10)
# ==========================================
TARGET_FILES = {
    "img-353.png", "img-43.png", "img-297.png", "img-25.png", "img-457.png",
    "img-588.png", "img-354.png", "img-577.png", "img-360.png", "img-271.png",
    "img-513.png", "img-218.png", "img-272.png", "img-193.png", "img-475.png",
    "img-359.png", "img-334.png", "img-352.png", "img-453.png", "img-178.png",
    "img-362.png", "img-93.png", "img-285.png", "img-288.png", "img-474.png",
    "img-37.png", "img-238.png", "img-531.png", "img-570.png", "img-247.png",
    "img-493.png", "img-106.png", "img-128.png", "img-379.png", "img-289.png",
    "img-146.png", "img-55.png", "img-95.png", "img-400.png", "img-259.png",
    "img-535.png", "img-208.png", "img-84.png", "img-505.png", "img-51.png",
    "img-224.png", "img-34.png", "img-429.png", "img-417.png", "img-254.png",
    "img-383.png", "img-317.png", "img-7.png", "img-340.png", "img-393.png",
    "img-594.png", "img-436.png", "img-468.png", "img-121.png", "img-127.png",
    "img-369.png", "img-327.png", "img-33.png", "img-401.png", "img-147.png",
    "img-335.png", "img-553.png", "img-301.png", "img-115.png", "img-258.png",
    "img-53.png", "img-198.png", "img-50.png", "img-70.png", "img-230.png",
    "img-550.png"
}

def get_args_parser():
    parser = argparse.ArgumentParser('Pano Segmentation Top-10 Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model', default='vit_huge_patch14', type=str)
    parser.add_argument('--pano_h', default=832, type=int)
    parser.add_argument('--pano_w', default=1664, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--nb_classes', default=8, type=int)
    parser.add_argument('--data_path', default='/media/data_hdd2/shanhefu/CVRG-Pano/test', type=str)
    parser.add_argument('--val_data_path', default='/media/data_hdd2/shanhefu/CVRG-Pano/test', type=str)
    parser.add_argument('--resume', default='/media/data_hdd1/shanhefu/outputs/finetune/segmentation_CVRG_Pano/sagementation_mask0.6-0.9_16*32_3.2e-3_decay_huge_2dcnn_16_CE_4_view/checkpoint-99.pth', help='Path to checkpoint')
    parser.add_argument('--output_dir', default='/media/data_hdd1/shanhefu/outputs/comparison/segmentation_visual_gap', help='Path to save results')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--input_size', default=None) 
    return parser

# CVRG-Pano 8类颜色映射
PALETTE = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
], dtype=np.uint8)

# ==========================================
# GPU 加速全景拼接函数
# ==========================================
def embed_patches_to_pano_gpu(patches_tensor, pano_h, pano_w, h_fov_deg, angles):
    device = patches_tensor.device
    N, C, H_patch, W_patch = patches_tensor.shape
    
    pano_canvas = torch.zeros((C, pano_h, pano_w), device=device, dtype=torch.float32)
    
    y_range = torch.arange(pano_h, device=device, dtype=torch.float32)
    x_range = torch.arange(pano_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    theta_omni = (2.0 * grid_x / (pano_w - 1) - 1.0) * torch.pi 
    phi_omni = (0.5 - grid_y / (pano_h - 1)) * torch.pi         
    
    cos_phi = torch.cos(phi_omni)
    pn = torch.stack([
        cos_phi * torch.cos(theta_omni),
        -cos_phi * torch.sin(theta_omni),
        torch.sin(phi_omni)
    ], dim=0)

    L = (W_patch / 2.0) / torch.tan(torch.deg2rad(torch.tensor(h_fov_deg / 2.0, device=device)))
    
    for i in range(N):
        theta_rad = torch.deg2rad(angles[i, 0])
        phi_rad = torch.deg2rad(angles[i, 1])
        
        nc = torch.tensor([
            torch.cos(phi_rad) * torch.cos(theta_rad),
            -torch.cos(phi_rad) * torch.sin(theta_rad),
            torch.sin(phi_rad)
        ], device=device).view(3, 1, 1)

        xn = torch.tensor([
            -torch.sin(theta_rad), -torch.cos(theta_rad), 0.0
        ], device=device).view(3, 1, 1)
        
        yn = torch.tensor([
            -torch.sin(phi_rad) * torch.cos(theta_rad),
            torch.sin(phi_rad) * torch.sin(theta_rad),
            torch.cos(phi_rad)
        ], device=device).view(3, 1, 1)
        
        cos_alpha = torch.sum(pn * nc, dim=0) 
        threshold = 2 * L / torch.sqrt(torch.tensor(W_patch**2 + H_patch**2 + 4*L**2, device=device))
        mask_fov = cos_alpha >= threshold
        
        if not mask_fov.any(): continue
        
        r_dist = L / (cos_alpha + 1e-8) 
        xp, yp = r_dist * torch.sum(pn * xn, dim=0), r_dist * torch.sum(pn * yn, dim=0)
        
        mask_rect = (xp > -W_patch/2.0) & (xp < W_patch/2.0) & (yp > -H_patch/2.0) & (yp < H_patch/2.0)
        final_mask = mask_fov & mask_rect
        if not final_mask.any(): continue
        
        c1 = torch.clamp((W_patch / 2.0 + xp[final_mask] - 0.5).round().long(), 0, W_patch - 1)
        r1 = torch.clamp((H_patch / 2.0 - yp[final_mask] - 0.5).round().long(), 0, H_patch - 1)
        
        pano_canvas[:, final_mask] = patches_tensor[i, :, r1, c1] 

    return pano_canvas

def compute_miou_single(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    if len(ious) == 0: return 0.0
    return np.mean(ious)

@torch.no_grad()
def run_top10_evaluation(data_loader, model, device, args):
    model.eval()
    results_list = []
    
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)
    h_fov_deg = 360.0 / (args.grid_height * 2)

    rgb_dir = os.path.join(args.val_data_path, 'rgb')
    if not os.path.exists(rgb_dir):
        rgb_dir = args.val_data_path
    img_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))])

    # ==========================================
    # Phase 1: 扫描并过滤目标列表中的图片
    # ==========================================
    print(f"--- Phase 1: Scanning test set (Targeting {len(TARGET_FILES)} specified images) ---")
    for batch_idx, (views, angles, targets) in enumerate(data_loader):
        views, angles, targets = views.to(device), angles.to(device), targets.to(device)
        B, N, C = views.shape[:3]
        
        batch_valid = False
        batch_fnames = []
        for b in range(B):
            global_idx = batch_idx * args.batch_size + b
            if global_idx < len(img_files):
                fname = img_files[global_idx]
                batch_fnames.append(fname)
                if fname in TARGET_FILES:
                    batch_valid = True
            else:
                batch_fnames.append("unknown")
                
        if not batch_valid:
            continue

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(views, angles)
            logits_upsampled = F.interpolate(
                logits.view(B*N, -1, logits.shape[-2], logits.shape[-1]),
                size=(patch_h, patch_w), mode='bilinear', align_corners=False
            )
            preds = logits_upsampled.argmax(dim=1).view(B, N, patch_h, patch_w)
        
        for b in range(B):
            fname = batch_fnames[b]
            if fname not in TARGET_FILES:
                continue
                
            sample_pred = preds[b].cpu().numpy()
            targets_flat = targets[b].view(N, 1, targets.shape[-2], targets.shape[-1]).float()
            sample_gt = F.interpolate(targets_flat, size=(patch_h, patch_w), mode='nearest').long().squeeze(1).cpu().numpy()
            
            miou = compute_miou_single(sample_pred, sample_gt, args.nb_classes)
            global_idx = batch_idx * args.batch_size + b
            results_list.append({'index': global_idx, 'fname': fname, 'miou': miou})
            print(f"Evaluated target file: {fname} | mIoU: {miou:.4f}")

    # ==========================================
    # Phase 2: 提取目标中的 Top 10 索引
    # ==========================================
    if len(results_list) == 0:
        print("Error: No target files matched in the dataset.")
        return
        
    df = pd.DataFrame(results_list)
    top_10 = df.sort_values(by='miou', ascending=False).head(10).reset_index(drop=True)
    target_indices = set(top_10['index'].values)
    
    print(f"\n--- Top 10 Samples Identified from Target List ---")
    print(top_10)

    # ==========================================
    # Phase 3: GPU 重建并使用带 _RankN 的命名
    # ==========================================
    print(f"\n--- Phase 3: Generating separate image files for Top 10 ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    rank_map = {row['index']: (idx+1, row['fname']) for idx, row in top_10.iterrows()}

    for batch_idx, (views, angles, targets) in enumerate(data_loader):
        B = views.size(0)
        for b in range(B):
            global_idx = batch_idx * args.batch_size + b
            
            if global_idx in target_indices:
                rank, fname = rank_map[global_idx]
                miou_val = top_10.loc[top_10['index'] == global_idx, 'miou'].values[0]
                print(f"Rendering Rank {rank:02d} (File: {fname}, mIoU: {miou_val:.4f})...")
                
                v_t = views[b:b+1].to(device)
                a_t = angles[b].to(device)
                t_t = targets[b:b+1].to(device)
                N = v_t.size(1)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(v_t, angles[b:b+1].to(device))
                    logits_upsampled = F.interpolate(
                        logits.view(N, -1, logits.shape[-2], logits.shape[-1]),
                        size=(patch_h, patch_w), mode='bilinear', align_corners=False
                    )
                    pred_tensor = logits_upsampled.argmax(dim=1).unsqueeze(1).float()
                
                t_flat = t_t.view(N, 1, t_t.shape[-2], t_t.shape[-1]).float()
                gt_tensor = F.interpolate(t_flat, size=(patch_h, patch_w), mode='nearest')
                rgb_patches = (v_t[0] * std + mean).clamp(0, 1)

                pano_rgb_t = embed_patches_to_pano_gpu(rgb_patches, args.pano_h, args.pano_w, h_fov_deg, a_t)
                pano_rgb_img = (pano_rgb_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                pano_pred_t = embed_patches_to_pano_gpu(pred_tensor, args.pano_h, args.pano_w, h_fov_deg, a_t)
                pano_gt_t = embed_patches_to_pano_gpu(gt_tensor, args.pano_h, args.pano_w, h_fov_deg, a_t)

                pano_pred_idx = pano_pred_t.squeeze(0).long().cpu().numpy()
                pano_gt_idx = pano_gt_t.squeeze(0).long().cpu().numpy()
                
                pano_pred_img = PALETTE[pano_pred_idx]
                pano_gt_img = PALETTE[pano_gt_idx]

                # ---------------- 独立文件夹保存 (添加 _RankN) ----------------
                # 提取去后缀的文件名，如 "img-353"
                base_name = os.path.splitext(fname)[0] 
                
                # 构建带有 _RankN 的文件夹名称，例如 "img-353.png_Rank1"
                folder_name = f"{fname}_Rank{rank}"
                save_folder = os.path.join(args.output_dir, folder_name)
                os.makedirs(save_folder, exist_ok=True)

                # 图片保存名也带上 _RankN
                Image.fromarray(pano_rgb_img).save(os.path.join(save_folder, f"{base_name}_Rank{rank}_RGB.png"))
                Image.fromarray(pano_gt_img).save(os.path.join(save_folder, f"{base_name}_Rank{rank}_GT.png"))
                Image.fromarray(pano_pred_img).save(os.path.join(save_folder, f"{base_name}_Rank{rank}_Predicted.png"))
                
                with open(os.path.join(save_folder, "info.txt"), "w") as info_file:
                    info_file.write(f"File Name: {fname}\n")
                    info_file.write(f"Global Index: {global_idx}\n")
                    info_file.write(f"Rank: {rank}\n")
                    info_file.write(f"mIoU: {miou_val:.4f}\n")
                
                target_indices.remove(global_idx)
                
        if not target_indices:
            break

    print(f"\n[✔] Folders generated successfully at: {args.output_dir}")

def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)
    real_patch_size = (patch_h, patch_w)
    args.input_size = real_patch_size

    dataset_val = build_segmentation_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False, shuffle=False
    )

    print(f"Creating model: {args.model}")
    model = models_segmentation.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=real_patch_size,
        patch_size=real_patch_size,
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w)
    )

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    run_top10_evaluation(data_loader_val, model, device, args)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)