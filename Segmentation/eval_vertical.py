import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import util.misc as misc
import util.odi_processing as odi

import Segmentation.models_segmentation_2dcnn as models_panomae

# ==============================================================================
# 1. Dataset
# ==============================================================================
class RawPanoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, 'rgb')
        self.mask_dir = os.path.join(root_dir, 'mask')
        valid_exts = {'.jpg', '.jpeg', '.png'}
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if os.path.splitext(f)[1].lower() in valid_exts])
        print(f"Found {len(self.filenames)} images for evaluation.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        try:
            mask = Image.open(os.path.join(self.mask_dir, os.path.splitext(fname)[0] + '.png'))
        except:
            mask = Image.new('L', img.size, 0)
        return np.array(img), np.array(mask), fname

# ==============================================================================
# 2. 核心评估引擎 (Vertical Version)
# ==============================================================================
@torch.no_grad()
def evaluate_vertical(dataloader, model, device, args, shift_ratio=0.0):
    header = f'Test Vertical Shift {int(shift_ratio*100)}%:'
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()

    # ODI 参数
    v_steps = args.grid_height
    u_steps = 2 * args.grid_height
    v_fov = 180.0 / v_steps
    h_fov = 360.0 / u_steps
    
    # Angle Centers
    v_centers = torch.linspace(90 - v_fov/2, -90 + v_fov/2, v_steps)
    u_centers = torch.linspace(-180, 180, u_steps + 1)[:-1]
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing='ij')
    angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    total_inter = torch.zeros(args.nb_classes, device=device)
    total_union = torch.zeros(args.nb_classes, device=device)
    total_target = torch.zeros(args.nb_classes, device=device)

    for img_np, mask_np, fname in metric_logger.log_every(dataloader, 10, header):
        # 取单张图
        img_np = img_np[0].numpy()  # (H, W, 3)
        mask_np = mask_np[0].numpy() # (H, W)
        
        # ==========================================================
        # [核心修改] 垂直滚动 (Axis 0)
        # ==========================================================
        if shift_ratio > 0:
            height = img_np.shape[0]
            shift_pixels = int(height * shift_ratio)
            img_np = np.roll(img_np, shift_pixels, axis=0)
            mask_np = np.roll(mask_np, shift_pixels, axis=0)

        # ODI 切片 (对移动后的图进行切片)
        raw_h = img_np.shape[0] // v_steps
        raw_w = img_np.shape[1] // u_steps
        if raw_h == raw_w: target_size = int(raw_h)
        else: target_size = (int(raw_w), int(raw_h))

        rgb_patches_list = odi.extract_patches_from_pano(
            pano_np=img_np, h_num=u_steps, v_num=v_steps,
            h_fov_deg=h_fov, v_fov_deg=v_fov, out_size=target_size
        )
        mask_patches_list = odi.extract_patches_from_pano(
            pano_np=mask_np, h_num=u_steps, v_num=v_steps,
            h_fov_deg=h_fov, v_fov_deg=v_fov, out_size=target_size
        )

        # To Tensor
        rgb_tensors = []
        mask_tensors = []
        for i in range(len(rgb_patches_list)):
            p_rgb = Image.fromarray(rgb_patches_list[i])
            p_rgb_tensor = normalize(transforms.ToTensor()(p_rgb))
            rgb_tensors.append(p_rgb_tensor)
            
            p_mask_np = mask_patches_list[i]
            if p_mask_np.ndim == 3: p_mask_np = p_mask_np[:,:,0]
            mask_tensors.append(torch.from_numpy(p_mask_np.astype(np.int64)))
            
        views = torch.stack(rgb_tensors).unsqueeze(0).to(device)
        targets = torch.stack(mask_tensors).unsqueeze(0).to(device)
        angles = angle_centers.unsqueeze(0).to(device)

        # 推理
        with torch.cuda.amp.autocast():
            logits = model(views, angles) 
            
            B, N, C, H, W = logits.shape
            th, tw = targets.shape[-2], targets.shape[-1]
            
            logits_upsampled = F.interpolate(
                logits.view(B*N, C, H, W), size=(th, tw), mode='bilinear', align_corners=False
            )
            targets_flat = targets.view(B*N, th, tw)
            
            pred_labels = logits_upsampled.argmax(dim=1)
            inter, union, target_cnt = compute_metrics(pred_labels, targets_flat, args.nb_classes)
            total_inter += inter
            total_union += union
            total_target += target_cnt

    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(total_inter, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_union, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_target, op=torch.distributed.ReduceOp.SUM)

    miou = (total_inter / (total_union + 1e-6)).mean().item()
    macc = (total_inter / (total_target + 1e-6)).mean().item()

    print(f'* [Vert {int(shift_ratio*100)}%] mIoU {miou:.4f} mAcc {macc:.4f}')
    return miou, macc

def compute_metrics(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    target_counts = torch.zeros(num_classes, device=pred.device)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds & target_inds).sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]
        target_counts[cls] = target_inds.sum()
    return intersection, union, target_counts

def get_args_parser():
    parser = argparse.ArgumentParser('PanoMAE Vertical Shift Eval', add_help=False)
    parser.add_argument('--model', default='vit_huge_patch14', type=str) 
    parser.add_argument('--grid_height', default=32, type=int) 
    parser.add_argument('--pano_h', default=832, type=int)
    parser.add_argument('--pano_w', default=1664, type=int)
    parser.add_argument('--nb_classes', default=8, type=int)
    parser.add_argument('--val_data_path', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser

def main(args):
    # Init CUDA explicitly
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

    # Dataset
    dataset = RawPanoDataset(root_dir=args.val_data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    # Model
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    patch_h = args.pano_h // v_steps
    patch_w = args.pano_w // u_steps
    real_patch_size = (patch_h, patch_w)
    
    print(f"Creating PanoMAE model: {args.model}")
    model = models_panomae.__dict__[args.model](
        grid_height=args.grid_height,
        output_size=(args.pano_h, args.pano_w),
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        img_size=real_patch_size, 
        patch_size=real_patch_size
    )

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Weight Interpolation
    if 'patch_embed.proj.weight' in state_dict:
        ckpt_w = state_dict['patch_embed.proj.weight']
        model_w = model.patch_embed.proj.weight
        if ckpt_w.shape != model_w.shape:
            print(f"[Auto-Fix] Resizing PatchEmbed weights from {ckpt_w.shape} to {model_w.shape}")
            ckpt_w = F.interpolate(ckpt_w, size=model_w.shape[2:], mode='bicubic', align_corners=False)
            state_dict['patch_embed.proj.weight'] = ckpt_w

    msg = model.load_state_dict(state_dict, strict=True)
    print("Loaded PanoMAE Checkpoint:", msg)

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Eval Loop
    shifts = [0.0, 0.25, 0.5, 0.75]
    results = []
    
    print(f"\nStart PanoMAE Vertical Shift Eval...")
    for s in shifts:
        miou, macc = evaluate_vertical(dataloader, model, device, args, shift_ratio=s)
        results.append((s, miou, macc))

    # Summary
    if misc.is_main_process():
        print("\n" + "="*40)
        print(" PanoMAE Vertical Shift Summary")
        print("="*40)
        print(f"{'Shift':<10} | {'mIoU':<10} | {'mAcc':<10}")
        print("-" * 36)
        avg_miou = 0
        for s, miou, macc in results:
            print(f"{int(s*100):<10}% | {miou:.4f}     | {macc:.4f}")
            avg_miou += miou
        print("-" * 36)
        print(f"{'Average':<10} | {avg_miou/len(results):.4f}")
        print("="*40)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)