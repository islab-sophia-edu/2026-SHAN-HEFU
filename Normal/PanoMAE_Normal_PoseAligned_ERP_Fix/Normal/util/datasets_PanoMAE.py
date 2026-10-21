import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import math
from pathlib import Path
import util.odi_processing 
import numpy as np

class VectorizedTensorRandomResizedCrop(torch.nn.Module):
    """
    A fully-vectorized, GPU-native PyTorch implementation of RandomResizedCrop.
    """
    def __init__(self, size, scale=(0.6, 1.0), ratio=(3./4., 4./3.)):
        super().__init__()
        self.size = (size, size)
        self.scale = scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def forward(self, img_batch):
        B, C, H, W = img_batch.shape
        device = img_batch.device
        
        target_areas = torch.rand(B, device=device) * (self.scale[1] - self.scale[0]) + self.scale[0]
        target_areas *= (H * W)
        
        log_ratios = torch.rand(B, device=device) * (self.log_ratio[1] - self.log_ratio[0]) + self.log_ratio[0]
        aspect_ratios = torch.exp(log_ratios)

        ws = torch.sqrt(target_areas * aspect_ratios).round().int()
        hs = torch.sqrt(target_areas / aspect_ratios).round().int()

        ws = torch.clamp(ws, min=1, max=W)
        hs = torch.clamp(hs, min=1, max=H)

        tops = torch.rand(B, device=device) * (H - hs)
        lefts = torch.rand(B, device=device) * (W - ws)
        
        scale_x = ws / W
        scale_y = hs / H
        trans_x = (2 * lefts + ws - W) / W
        trans_y = (2 * tops + hs - H) / H
        
        zeros = torch.zeros(B, device=device)
        theta = torch.stack([
            scale_x, zeros,   trans_x,
            zeros,   scale_y, trans_y
        ], dim=1).view(B, 2, 3)

        grid = F.affine_grid(theta, (B, C, self.size[0], self.size[1]), align_corners=False)
        return F.grid_sample(img_batch, grid, mode='bicubic', padding_mode='reflection', align_corners=False)


class PanoramicDataset(Dataset):
    """
    Dataset with Input Resizing and GPU-accelerated multiscale augmentation.
    """
    def __init__(self, root_dir: str, grid_height: int = 4, img_size: int = 224,
                 pano_h: int = 512, pano_w: int = 1024, # <-- NEW
                 multiscale_sampling: bool = False, 
                 use_horizontal_roll: bool = False,
                 device: str = None):
        
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.img_size = img_size
        
        # Store expected dimensions
        self.pano_h = pano_h
        self.pano_w = pano_w

        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"PanoramicDataset initialized. Target Pano Size: {self.pano_w}x{self.pano_h}. Patch Size: {self.img_size}")

        self.v_steps, self.u_steps = grid_height, 2 * grid_height
        self.v_fov, self.h_fov = 180.0 / self.v_steps, 360.0 / self.u_steps
        v_step_size = 180.0 / self.v_steps
        v_centers_deg = torch.linspace(90 - v_step_size / 2, -90 + v_step_size / 2, self.v_steps)
        u_centers_deg = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers_deg, u_centers_deg, indexing='ij')
        
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)

        self.use_multiscale = multiscale_sampling
        if self.use_multiscale:
            self.transform_aug = VectorizedTensorRandomResizedCrop(img_size, scale=(0.6, 1.0))
        
        self.use_horizontal_roll = use_horizontal_roll
        self.transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            pano_pil = Image.open(img_path).convert('RGB')
            
            # --- MODIFICATION: RESIZE INPUT ---
            # 强制调整为 args 中指定的 pano_w 和 pano_h
            if pano_pil.size != (self.pano_w, self.pano_h):
                pano_pil = pano_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)
            # ----------------------------------

            pano_np = np.array(pano_pil)
            
            # Get current width (which should be self.pano_w now)
            pano_w = pano_np.shape[1] 
            
            if self.use_horizontal_roll:
                shift_px = torch.randint(0, pano_w, (1,)).item()
                pano_np = np.roll(pano_np, shift_px, axis=1)

            original_size = torch.tensor(pano_pil.size, dtype=torch.long)
            
            # odi_processing: Extract Patches based on the resized image
            patches_np_list = util.odi_processing.extract_patches_from_pano(
                pano_np=pano_np, 
                h_num=self.u_steps, 
                v_num=self.v_steps,
                h_fov_deg=self.h_fov, 
                v_fov_deg=self.v_fov, 
                out_size=self.img_size # Calculated from main.py (pano_h // grid_h)
            )
            patches = torch.stack([transforms.ToTensor()(p) for p in patches_np_list])

        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            num_patches = self.u_steps * self.v_steps
            return (
                torch.empty(num_patches, 3, self.img_size, self.img_size),
                torch.empty(num_patches, 2),
                torch.empty(2) 
            )

        patches = patches.to(self.device)
        
        if self.use_multiscale:
            patches = self.transform_aug(patches)

        final_patches = self.transform_norm(patches)
        angles = self.angle_centers
        
        return final_patches, angles, original_size