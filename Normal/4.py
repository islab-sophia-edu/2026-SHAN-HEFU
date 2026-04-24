import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

# Dynamic Import
try:
    from models_normal import vit_huge_patch14
except ImportError:
    print("Error: models_normal.py not found.")
    exit(1)

# =========================================================
# 1. Dataset
# =========================================================
class DiagnosticDataset(Dataset):
    def __init__(self, root_dir, pano_h, pano_w):
        self.root_dir = root_dir
        self.pano_h = pano_h
        self.pano_w = pano_w
        self.filenames = []
        
        # Area scanning
        areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6']
        print(f"Scanning dataset in {root_dir}...")
        
        for area in areas:
            rgb_dir = os.path.join(root_dir, area, 'pano', 'rgb')
            # Handle naming variations
            norm_dir = os.path.join(root_dir, area, 'pano', 'normal')
            if not os.path.exists(norm_dir):
                norm_dir = os.path.join(root_dir, area, 'pano', 'normals')
            
            if not os.path.exists(rgb_dir) or not os.path.exists(norm_dir):
                continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            for rgb_path in rgb_files:
                f_name = os.path.basename(rgb_path)
                # Try matching normal file
                n_name = f_name.replace('_rgb.png', '_normals.png')
                n_path = os.path.join(norm_dir, n_name)
                
                if not os.path.exists(n_path):
                     n_path = os.path.join(norm_dir, f_name.replace('_rgb.png', '_normal.png'))
                
                if os.path.exists(n_path):
                    self.filenames.append({'rgb': rgb_path, 'normal': n_path, 'id': f"{area}/{f_name}"})
        
        print(f"Found {len(self.filenames)} valid RGB-Normal pairs.")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.filenames[idx]
        
        # Load RGB
        img_pil = Image.open(data['rgb']).convert('RGB')
        img_pil = img_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)
        img_tensor = self.normalize(transforms.ToTensor()(img_pil))
        
        # Load Normal (GT)
        norm_pil = Image.open(data['normal']).convert('RGB')
        # Nearest to preserve edges/values
        norm_pil = norm_pil.resize((self.pano_w, self.pano_h), Image.NEAREST) 
        norm_np = np.array(norm_pil).astype(np.float32) / 255.0 
        norm_tensor = torch.from_numpy(norm_np).permute(2, 0, 1) # [3, H, W]
        
        # Map [0, 1] -> [-1, 1]
        norm_physics = norm_tensor * 2.0 - 1.0
        
        return img_tensor, norm_physics, data['id']

# =========================================================
# 2. Physics & Handedness Analyzer
# =========================================================
def analyze_vector_field(tensor_bhw, name="Data"):
    """
    Analyzes Coordinate System (Up-Axis) and Handedness (RHS vs LHS).
    tensor_bhw: [B, 3, H, W], values in [-1, 1]
    """
    B, C, H, W = tensor_bhw.shape
    
    # --- ROI Extraction ---
    # 1. Floor (Bottom 10%) -> Defines UP vector
    floor_region = tensor_bhw[:, :, int(H*0.9):, :]
    floor_mean = F.normalize(floor_region.mean(dim=(2, 3)), p=2, dim=1) # [B, 3]

    # 2. Center Wall (Center crop) -> Defines -FORWARD vector (Normal points back to camera)
    # Taking a small crop in the exact center
    center_region = tensor_bhw[:, :, int(H*0.45):int(H*0.55), int(W*0.48):int(W*0.52)]
    center_mean = F.normalize(center_region.mean(dim=(2, 3)), p=2, dim=1)

    # 3. Right Wall (W/4 crop) -> Defines -RIGHT vector (Normal points left to camera)
    # In equirectangular, W/4 corresponds to 90 degrees right
    right_region = tensor_bhw[:, :, int(H*0.45):int(H*0.55), int(W*0.23):int(W*0.27)]
    right_wall_mean = F.normalize(right_region.mean(dim=(2, 3)), p=2, dim=1)

    results = []
    axis_names = ['X', 'Y', 'Z']

    for i in range(B):
        # --- Axis Analysis ---
        f_vec = floor_mean[i]
        abs_f = f_vec.abs()
        up_idx = torch.argmax(abs_f).item()
        up_sign = 1 if f_vec[up_idx] > 0 else -1
        
        coord_sys = "Unknown"
        if up_idx == 1: coord_sys = "Y-Up"
        elif up_idx == 2: coord_sys = "Z-Up"
        elif up_idx == 0: coord_sys = "X-Up"

        # --- Handedness Analysis ---
        # Hypothesis:
        # Vector_Up ~ Floor_Normal
        # Vector_Fwd ~ -(Center_Wall_Normal)  [Wall faces camera, so normal is back]
        # Vector_Right ~ -(Right_Wall_Normal) [Wall faces left, so normal is left]
        
        vec_up = f_vec
        vec_fwd = -center_mean[i]
        vec_right_obs = -right_wall_mean[i] # Observed Right Axis
        
        # Standard Cross Product Check (Right-Hand Rule)
        # In RHS: Cross(Forward, Up) should align with Right
        # In LHS: Cross(Forward, Up) should align with -Right (or Cross(Up, Fwd) = Right)
        
        cross_res = torch.cross(vec_fwd, vec_up)
        
        # Check alignment with observed right vector
        dot_val = torch.dot(cross_res, vec_right_obs).item()
        
        # If dot > 0, Cross(F, U) == R  -> Right Handed System
        # If dot < 0, Cross(F, U) == -R -> Left Handed System
        handedness = "RHS" if dot_val > 0 else "LHS"
        
        results.append({
            "coord_type": coord_sys,
            "handedness": handedness,
            "up_axis_str": f"{['-', '+'][int(up_sign>0)]}{axis_names[up_idx]}",
            "floor_vec": f_vec.tolist(),
            "desc": f"{coord_sys} ({handedness})"
        })
        
    return results

# =========================================================
# 3. Main Routine
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', default='', required=True, help='Path to dataset')
    parser.add_argument('--pano_h', default=512, type=int)
    parser.add_argument('--pano_w', default=1024, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Diagnostics on: {device}")

    # --- Load Model ---
    print(f"\n[Step 1] Loading Model...")
    # Initialize with patch logic estimates
    model = vit_huge_patch14(img_size=(args.pano_h//16, args.pano_w//32), 
                             patch_size=(args.pano_h//16, args.pano_w//32), 
                             in_chans=3, grid_height=16, 
                             output_size=(args.pano_h, args.pano_w))
    
    print(f"Loading Weights: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Clean state dict keys
    new_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_dict, strict=False)
    model.to(device).eval()

    # --- Load Data ---
    print(f"\n[Step 2] Init DataLoader...")
    dataset = DiagnosticDataset(args.data_path, args.pano_h, args.pano_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Analysis Loop ---
    print(f"\n[Step 3] Analyzing Geometry (Up-Axis & Handedness)...")
    
    stats = {
        "GT": {"Y-Up": 0, "Z-Up": 0, "X-Up": 0, "RHS": 0, "LHS": 0},
        "Pred": {"Y-Up": 0, "Z-Up": 0, "X-Up": 0, "RHS": 0, "LHS": 0}
    }
    
    limit_batches = 20000 # Enough samples for statistical significance
    
    with torch.no_grad():
        for i, (imgs, gt_norms, ids) in enumerate(tqdm(loader)):
            if i >= limit_batches: break
            
            imgs = imgs.to(device)
            gt_norms = gt_norms.to(device)
            
            # 1. Analyze Ground Truth
            gt_res = analyze_vector_field(gt_norms, "GT")
            
            # 2. Run Inference
            # Construct Dummy Angles for ViT (B, N, 2)
            B = imgs.shape[0]
            grid_h, grid_w = 16, 32
            u = torch.linspace(-180, 180, grid_w+1)[:-1]
            v = torch.linspace(90, -90, grid_h)
            v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
            angles = torch.stack([u_grid, v_grid], dim=-1).flatten(0, 1).unsqueeze(0).expand(B, -1, -1).to(device)
            
            # Patchify Input
            p_h, p_w = args.pano_h // grid_h, args.pano_w // grid_w
            patches = imgs.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
            patches = patches.contiguous().view(B, 3, grid_h, grid_w, p_h, p_w)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3, p_h, p_w).contiguous()
            
            # Forward
            preds_patches = model(patches, angles) 
            
            # Un-patchify
            preds_patches = preds_patches.view(B, grid_h, grid_w, 3, p_h, p_w)
            preds_patches = preds_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
            preds_full = preds_patches.view(B, 3, args.pano_h, args.pano_w)
            
            # 3. Analyze Prediction
            pred_res = analyze_vector_field(preds_full, "Pred")
            
            # 4. Update Stats
            for b in range(B):
                # GT Stats
                g_sys = gt_res[b]['coord_type']
                g_hand = gt_res[b]['handedness']
                stats["GT"][g_sys] = stats["GT"].get(g_sys, 0) + 1
                stats["GT"][g_hand] = stats["GT"].get(g_hand, 0) + 1
                
                # Pred Stats
                p_sys = pred_res[b]['coord_type']
                p_hand = pred_res[b]['handedness']
                stats["Pred"][p_sys] = stats["Pred"].get(p_sys, 0) + 1
                stats["Pred"][p_hand] = stats["Pred"].get(p_hand, 0) + 1
                
                # Verbose Mismatch Print (First Batch only)
                if i == 0 and b < 2: 
                    print(f"\n--- ID: {ids[b]} ---")
                    print(f"GT   : {gt_res[b]['desc']} | Up-Vec: {['{:.2f}'.format(x) for x in gt_res[b]['floor_vec']]}")
                    print(f"Model: {pred_res[b]['desc']} | Up-Vec: {['{:.2f}'.format(x) for x in pred_res[b]['floor_vec']]}")

    # --- Report ---
    print("\n" + "="*60)
    print("FINAL COORDINATE & HANDEDNESS REPORT")
    print("="*60)
    
    total = sum(stats["GT"].values()) // 2 # Divided by 2 because we count sys and hand separately roughly
    
    # Determine Dominants
    dom_gt_sys = max(["Y-Up", "Z-Up", "X-Up"], key=lambda k: stats["GT"][k])
    dom_gt_hand = max(["RHS", "LHS"], key=lambda k: stats["GT"][k])
    
    dom_pred_sys = max(["Y-Up", "Z-Up", "X-Up"], key=lambda k: stats["Pred"][k])
    dom_pred_hand = max(["RHS", "LHS"], key=lambda k: stats["Pred"][k])

    print(f"[Dataset Reality]: {dom_gt_sys}, {dom_gt_hand}")
    print(f"  - Details: {stats['GT']}")
    
    print(f"[Model Preference]: {dom_pred_sys}, {dom_pred_hand}")
    print(f"  - Details: {stats['Pred']}")
    
    print("-" * 60)
    
    match_sys = (dom_gt_sys == dom_pred_sys)
    match_hand = (dom_gt_hand == dom_pred_hand)
    
    if match_sys and match_hand:
        print("✅ PASS: Dataset and Model match perfectly.")
    else:
        print("❌ FAIL: Coordinate Mismatch Detected.")
        if not match_sys:
            print(f"   > AXIS ERROR: Dataset is {dom_gt_sys}, Model needs {dom_pred_sys}.")
        if not match_hand:
            print(f"   > HANDEDNESS ERROR: Dataset is {dom_gt_hand}, Model needs {dom_pred_hand}.")
            print("   > ACTION: You need to flip one axis (e.g., x = -x) to fix handedness.")

if __name__ == '__main__':
    main()