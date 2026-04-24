import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 确保这些文件在你的路径中
import util.odi_processing_gpu as odi_helper
from dataset_stanford_normal_6ch import build_normal_dataset

# 导入模型
try:
    from models_normal_cnn import vit_huge_patch14
except ImportError:
    print("Error: Could not import 'vit_huge_patch14'. Check file existence.")
    sys.exit(1)

# ==============================================================================
# 1. 逆变换逻辑 (与 Dataset Candidate 4 保持严格一致)
# ==============================================================================
class VisCameraPrm:
    def __init__(self, camera_angle, image_plane_size, view_angle):
        self.camera_angle = camera_angle
        L = (image_plane_size[0] / 2.0) / np.tan(view_angle[0] / 2.0)
        self.L = L
        ca = camera_angle
        
        # Standard ODI definitions
        self.nc = np.array([
            np.cos(ca[1]) * np.cos(ca[0]), 
            -np.cos(ca[1]) * np.sin(ca[0]), 
            np.sin(ca[1])
        ])
        
        # [SPATIAL FIX] Flipped XN (Match Dataset Logic)
        self.xn = -1.0 * np.array([
            -np.sin(ca[0]), 
            -np.cos(ca[0]),
            0
        ])
        
        self.yn = np.array([
            -np.sin(ca[1]) * np.cos(ca[0]), 
            np.sin(ca[1]) * np.sin(ca[0]),
            np.cos(ca[1])
        ])

def get_cam_rotation_matrix_candidate_4(prm):
    # Match Dataset Logic
    odi_nc, odi_xn, odi_yn = prm.nc, prm.xn, prm.yn
    
    # 1. Map to Stanford (Direct)
    cam_right_raw = np.array([ -odi_xn[1], odi_xn[2], odi_xn[0] ])
    cam_down      = np.array([ -odi_yn[1], odi_yn[2], odi_yn[0] ])
    cam_fwd       = np.array([ -odi_nc[1], odi_nc[2], odi_nc[0] ])
    
    # 2. Invert Right (Candidate 4)
    cam_right = -1.0 * cam_right_raw
    
    return np.stack([cam_right, cam_down, cam_fwd], axis=0)

def inverse_rotate_batch(patches_cam, angles, h_fov, v_fov):
    """
    将 (B, N, 3, H, W) 的 Camera Space 法线转回 World Space
    """
    B, N, C, H, W = patches_cam.shape
    device = patches_cam.device
    
    patches_world_list = []
    
    patches_np = patches_cam.permute(0, 1, 3, 4, 2).cpu().numpy() # (B, N, H, W, 3)
    angles_np = angles.cpu().numpy() # (B, N, 2) degrees
    
    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)
    
    for b in range(B):
        batch_list = []
        for n in range(N):
            theta = np.radians(angles_np[b, n, 0])
            phi = np.radians(angles_np[b, n, 1])
            
            prm = VisCameraPrm([theta, phi], [W, H], [h_fov_rad, v_fov_rad])
            R = get_cam_rotation_matrix_candidate_4(prm) # (3, 3)
            
            # Inverse: v_world = v_cam @ (R^T)^-1 = v_cam @ R
            p_cam = patches_np[b, n] 
            flat_cam = p_cam.reshape(-1, 3)
            flat_world = np.dot(flat_cam, R) 
            p_world = flat_world.reshape(H, W, 3)
            
            # Normalize
            norm = np.linalg.norm(p_world, axis=2, keepdims=True)
            p_world = p_world / (norm + 1e-8)
            
            batch_list.append(p_world)
        patches_world_list.append(np.stack(batch_list))
        
    return torch.from_numpy(np.array(patches_world_list)).permute(0, 1, 4, 2, 3).float().to(device)

# ==============================================================================
# 2. 误差计算工具
# ==============================================================================
def calc_angular_error(n1, n2, valid_mask=None):
    """
    n1, n2: (3, H, W) or (1, 3, H, W), range [-1, 1]
    Return: Mean Angular Error (Degrees)
    """
    sim = torch.sum(n1 * n2, dim=-3, keepdim=True) 
    sim = torch.clamp(sim, -1.0, 1.0)
    angles = torch.acos(sim) * (180 / np.pi)
    
    if valid_mask is not None:
        valid_angles = angles[valid_mask > 0.5]
        if valid_angles.numel() == 0: return 0.0
        return valid_angles.mean().item()
    else:
        return angles.mean().item()

def vis_norm(tensor):
    return torch.clamp((tensor + 1)/2.0, 0, 1)

# ==============================================================================
# 3. 主程序
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--pano_h', default=512, type=int)
    parser.add_argument('--pano_w', default=1024, type=int)
    parser.add_argument('--grid_height', default=16, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', default='vit_huge_patch14', type=str)
    parser.add_argument('--resume', default='', type=str, required=True)
    parser.add_argument('--val_data_path', default='/home/shanhefu/Stanford2D3D', type=str)
    parser.add_argument('--output_dir', default='./vis_compare_output', type=str)
    parser.add_argument('--num_samples', default=5, type=int)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    u_steps = 2 * args.grid_height
    v_steps = args.grid_height
    h_fov = 360.0 / u_steps
    v_fov = 180.0 / v_steps
    img_size = args.pano_h // args.grid_height

    # Dataset (Val)
    dataset = build_normal_dataset(is_train=False, args=args)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model
    print(f"Loading model from {args.resume}")
    model = globals()[args.model](
        img_size=img_size, patch_size=img_size, in_chans=6, num_classes=3,
        grid_height=args.grid_height, output_size=(args.pano_h, args.pano_w)
    )
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    model.to(device).eval()

    print(f"Start Visualization... ({len(dataset)} samples)")

    for i, batch in enumerate(data_loader):
        if i >= args.num_samples: break
        
        # 1. Run Model
        # targets: (B, N, 3, H, W) in Camera Space
        views, angles, targets, masks = batch
        views, angles, targets = views.to(device), angles.to(device), targets.to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(views, angles) 
        preds = preds.float()
        
        if preds.shape[-1] != img_size:
            preds = F.interpolate(preds.view(-1, 3, preds.shape[-1], preds.shape[-1]), 
                                size=(img_size, img_size), mode='bilinear').view(1, -1, 3, img_size, img_size)

        # 2. Inverse Rotate (Camera -> World)
        print(f"[{i}] Inverse Rotating Patches...")
        preds_world = inverse_rotate_batch(preds, angles, h_fov, v_fov)
        targets_world = inverse_rotate_batch(targets, angles, h_fov, v_fov)
        
        # 3. Stitch
        # RGB Input (Denormalize)
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        patches_rgb = torch.clamp(views[0, :, :3] * rgb_std + rgb_mean, 0, 1)
        
        pano_rgb = odi_helper.embed_patches_to_pano_gpu(
            patches_rgb, args.pano_h, args.pano_w, u_steps, v_steps, h_fov, v_fov, angles[0]
        )
        
        pano_pred_world = odi_helper.embed_patches_to_pano_gpu(
            preds_world[0], args.pano_h, args.pano_w, u_steps, v_steps, h_fov, v_fov, angles[0]
        )
        
        # Target World (Dataset -> Inverse -> Stitch)
        pano_target_world = odi_helper.embed_patches_to_pano_gpu(
            targets_world[0], args.pano_h, args.pano_w, u_steps, v_steps, h_fov, v_fov, angles[0]
        )
        
        # 4. Load Original GT File
        gt_path = dataset.filenames[i]['normal']
        gt_pil = Image.open(gt_path).convert('RGB').resize((args.pano_w, args.pano_h), Image.NEAREST)
        gt_np = np.array(gt_pil).astype(np.float32)
        
        # [CRITICAL UPDATE] Flip GT to match our "Flipped World" logic
        gt_np = np.fliplr(gt_np).copy()
        
        # To Tensor [-1, 1]
        gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).to(device)
        gt_tensor = (gt_tensor / 255.0) * 2.0 - 1.0
        gt_tensor = F.normalize(gt_tensor, dim=0, p=2) 
        
        # 5. Compute Errors
        # Stitching consistency check
        stitch_mask = (torch.norm(pano_target_world, dim=0, keepdim=True) > 0.1).float()
        
        # Error A: Consistency (Reconstructed Target vs Loaded Flipped GT) -> Should be 0
        err_stitch = calc_angular_error(pano_target_world, gt_tensor, stitch_mask)
        
        # Error B: Model Performance (Prediction vs Loaded Flipped GT)
        err_pred = calc_angular_error(pano_pred_world, gt_tensor, stitch_mask)
        
        # 6. Visualization
        img1 = pano_rgb
        img2 = vis_norm(pano_pred_world)
        img3 = vis_norm(pano_target_world)
        img4 = vis_norm(gt_tensor)
        
        combined = torch.cat([img1, img2, img3, img4], dim=1) 
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        axes[0].imshow(img1.permute(1,2,0).cpu().numpy())
        axes[0].set_title(f"1. Input RGB (Stitched)\nFile: {os.path.basename(gt_path)}")
        axes[0].axis('off')
        
        axes[1].imshow(img2.permute(1,2,0).cpu().numpy())
        axes[1].set_title(f"2. Prediction (World)\nPred Error: {err_pred:.2f} deg")
        axes[1].axis('off')
        
        axes[2].imshow(img3.permute(1,2,0).cpu().numpy())
        axes[2].set_title(f"3. Stitched Target (Recovered)\nConsistency Error: {err_stitch:.2f} deg (Must be low)")
        axes[2].axis('off')
        
        axes[3].imshow(img4.permute(1,2,0).cpu().numpy())
        axes[3].set_title("4. Original GT File (Flipped LR to match World definition)")
        axes[3].axis('off')
        
        out_file = os.path.join(args.output_dir, f"compare_{i}.png")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        
        print(f"Saved {out_file} | Consistency: {err_stitch:.4f} | Pred Error: {err_pred:.4f}")

    print("Comparison Done.")

if __name__ == '__main__':
    main()