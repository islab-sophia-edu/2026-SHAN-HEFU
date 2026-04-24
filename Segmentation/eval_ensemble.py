import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import util.misc as misc
import util.odi_processing as odi
import Segmentation.models_segmentation_2dcnn as models_panomae
import models_segmentation_baseline as models_baseline

# =========================================================
# 融合配置
# =========================================================
LARGE_CLASSES = [1, 2, 4, 5]  
WEIGHT_BASELINE = 0.5         
WEIGHT_PANOMAE = 0.5
# =========================================================

class RawPanoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        import os
        self.img_dir = os.path.join(root_dir, 'rgb')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        import os
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        try:
            mask_name = os.path.splitext(fname)[0] + '.png'
            mask = Image.open(os.path.join(self.mask_dir, mask_name))
        except:
            mask = Image.new('L', img.size, 0)
        return np.array(img), np.array(mask)

@torch.no_grad()
def evaluate_ensemble(dataloader, model_base, model_pano, device, args):
    model_base.eval()
    model_pano.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Reconstruct Ensemble:'
    
    total_inter = torch.zeros(args.nb_classes, device=device)
    total_union = torch.zeros(args.nb_classes, device=device)
    
    hard_classes = [c for c in range(args.nb_classes) if c not in LARGE_CLASSES]
    
    # ODI 坐标参数
    v_steps = args.grid_height
    u_steps = 2 * args.grid_height
    v_fov = 180.0 / v_steps
    h_fov = 360.0 / u_steps
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for img_np, mask_np in metric_logger.log_every(dataloader, 10, header):
        # 准备数据
        img_pil = Image.fromarray(img_np[0].numpy())
        target = torch.from_numpy(mask_np[0].numpy().astype(np.int64)).to(device)
        
        # ======================================================
        # Branch A: Baseline (ERP Input -> ERP Output)
        # ======================================================
        img_tensor_base = normalize(transforms.ToTensor()(img_pil)).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            logits_base = model_base(img_tensor_base) # (1, C, H, W)
            
            # 对齐尺寸
            if logits_base.shape[-2:] != target.shape[-2:]:
                logits_base = F.interpolate(logits_base, size=target.shape[-2:], mode='bilinear', align_corners=False)

        # ======================================================
        # Branch B: PanoMAE (ODI Input -> Patches -> Reconstruction)
        # ======================================================
        # 1. 准备 ODI 输入
        raw_h = img_np[0].shape[0] // v_steps
        raw_w = img_np[0].shape[1] // u_steps
        target_size = int(raw_h) if raw_h == raw_w else (int(raw_w), int(raw_h))
        
        rgb_patches = odi.extract_patches_from_pano(
            pano_np=img_np[0].numpy(), h_num=u_steps, v_num=v_steps,
            h_fov_deg=h_fov, v_fov_deg=v_fov, out_size=target_size
        )
        rgb_tensors = [normalize(transforms.ToTensor()(Image.fromarray(p))) for p in rgb_patches]
        views = torch.stack(rgb_tensors).unsqueeze(0).to(device)
        angles = angle_centers.unsqueeze(0).to(device)
        
        with torch.amp.autocast('cuda'):
            # 输出: (1, N, C, Ph, Pw)
            logits_pano_patches = model_pano(views, angles) 
            
            # ---------------------------------------------------
            # [核心修复] 逆向重组：把 Patch 拼回全景图 (ERP)
            # ---------------------------------------------------
            # PanoMAE forward 中的操作是: 
            # 1. view(B, C, GridH, Ph, GridW, Pw)
            # 2. permute(0, 2, 4, 1, 3, 5) -> (B, GridH, GridW, C, Ph, Pw)
            # 3. view(B, N, C, Ph, Pw)
            # 我们现在要倒着做回去：
            
            B, N, C, Ph, Pw = logits_pano_patches.shape
            GridH = args.grid_height
            GridW = 2 * args.grid_height
            
            # 1. 拆分 N -> (GridH, GridW)
            # shape: (B, GridH, GridW, C, Ph, Pw)
            x = logits_pano_patches.view(B, GridH, GridW, C, Ph, Pw)
            
            # 2. 逆向 Permute
            # 目标: (B, C, GridH, Ph, GridW, Pw)
            # 对应原索引: (0, 3, 1, 4, 2, 5)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            
            # 3. 合并维度 -> 全景图
            # shape: (B, C, GridH * Ph, GridW * Pw)
            logits_pano_erp = x.view(B, C, GridH * Ph, GridW * Pw)
            
            # 对齐尺寸
            if logits_pano_erp.shape[-2:] != target.shape[-2:]:
                logits_pano_erp = F.interpolate(logits_pano_erp, size=target.shape[-2:], mode='bilinear', align_corners=False)

        # ======================================================
        # Fusion & Stats
        # ======================================================
        # 现在的 logits_base 和 logits_pano_erp 是像素级对齐的！
        final_logits = (logits_base * WEIGHT_BASELINE) + (logits_pano_erp * WEIGHT_PANOMAE)
        pred_labels = final_logits.argmax(dim=1).view(-1)
        targets_flat = target.view(-1)

        for cls in range(args.nb_classes):
            pred_inds = (pred_labels == cls)
            target_inds = (targets_flat == cls)
            # 转 float 防止溢出
            inter = (pred_inds & target_inds).float().sum()
            union = (pred_inds | target_inds).float().sum()
            
            total_inter[cls] += inter
            total_union[cls] += union

    # --- Print ---
    iou_list = []
    hard_iou_list = []
    
    if misc.is_main_process():
        print("\n" + "="*40)
        print(" RECONSTRUCT ENSEMBLE RESULTS")
        print("="*40)
        
        for cls in range(args.nb_classes):
            iou = total_inter[cls] / (total_union[cls] + 1e-6)
            iou_list.append(iou.item())
            type_str = "HARD" if cls in hard_classes else "Large"
            print(f"Class {cls:<3}: IoU = {iou:.4f} [{type_str}]")
            
            if cls in hard_classes:
                hard_iou_list.append(iou.item())

        miou = sum(iou_list) / len(iou_list)
        hard_miou = sum(hard_iou_list) / len(hard_iou_list) if hard_iou_list else 0
        
        print("-" * 40)
        print(f"Global mIoU: {miou:.4f}")
        print(f"Hard mIoU  : {hard_miou:.4f}")
        print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', required=True)
    parser.add_argument('--ckpt_base', required=True)
    parser.add_argument('--ckpt_pano', required=True)
    
    parser.add_argument('--model', default='vit_huge_patch14')
    parser.add_argument('--nb_classes', default=8, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--pano_h', default=832, type=int)
    parser.add_argument('--pano_w', default=1664, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input_size', default=[832, 1664], nargs=2, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    
    args = parser.parse_args()
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # 1. Load Baseline
    print("Loading Baseline...")
    model_kwargs = {} 
    model_base = models_baseline.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=tuple(args.input_size),
        patch_size=args.patch_size,
        **model_kwargs
    )
    ckpt_b = torch.load(args.ckpt_base, map_location='cpu', weights_only=False)
    state_dict_b = ckpt_b['model'] if 'model' in ckpt_b else ckpt_b
    model_base.load_state_dict(state_dict_b, strict=False)
    model_base.to(device)

    # 2. Load PanoMAE
    print("Loading PanoMAE...")
    patch_h = args.pano_h // args.grid_height
    patch_w = args.pano_w // (2 * args.grid_height)
    model_pano = models_panomae.__dict__[args.model](
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        num_classes=args.nb_classes,
        img_size=(patch_h, patch_w),
        patch_size=(patch_h, patch_w)
    )
    ckpt_p = torch.load(args.ckpt_pano, map_location='cpu', weights_only=False)
    state_dict_p = ckpt_p['model'] if 'model' in ckpt_p else ckpt_p
    
    if 'head.weight' in state_dict_p and not hasattr(model_pano, 'head'):
        print("Remapping head -> final_conv for PanoMAE")
        state_dict_p['final_conv.weight'] = state_dict_p.pop('head.weight')
        state_dict_p['final_conv.bias'] = state_dict_p.pop('head.bias')
    
    if 'patch_embed.proj.weight' in state_dict_p:
        p_w = state_dict_p['patch_embed.proj.weight']
        if p_w.shape != model_pano.patch_embed.proj.weight.shape:
             print("Resizing PanoMAE patch_embed...")
             state_dict_p['patch_embed.proj.weight'] = F.interpolate(p_w, size=model_pano.patch_embed.proj.weight.shape[2:])

    model_pano.load_state_dict(state_dict_p, strict=False)
    model_pano.to(device)

    dataset = RawPanoDataset(args.val_data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    evaluate_ensemble(dataloader, model_base, model_pano, device, args)

if __name__ == '__main__':
    main()