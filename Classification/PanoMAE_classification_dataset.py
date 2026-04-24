import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import util.odi_processing
from typing import Dict, List, Optional

class StrictImageFolder(ImageFolder):
    """
    1. Filters out 'others' class automatically.
    2. Enforces a specific class-to-index mapping (solving 31 vs 25 mismatch).
    """
    def __init__(self, root, predefined_class_to_idx: Optional[Dict[str, int]] = None, **kwargs):
        self.predefined_class_to_idx = predefined_class_to_idx
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        # 1. Scan directory for classes
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        # 2. Filter out 'others' explicitly (Teacher's advice)
        if 'others' in classes:
            classes.remove('others')
            
        # 3. Handle Training Mode (Create Mapping)
        if self.predefined_class_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx
            
        # 4. Handle Validation Mode (Enforce Mapping)
        else:
            # Only keep classes that exist in BOTH the directory AND the predefined mapping
            valid_classes = [c for c in classes if c in self.predefined_class_to_idx]
            
            # CRITICAL: Use the indices from Training set!
            class_to_idx = {c: self.predefined_class_to_idx[c] for c in valid_classes}
            
            return valid_classes, class_to_idx

class PanoClassificationDataset(Dataset):
    def __init__(self, root_dir, grid_height=4, img_size=128, transform=None, is_train=True, device='cuda', class_to_idx=None):
        self.root_dir = root_dir
        
        # Use Custom ImageFolder
        self.image_folder = StrictImageFolder(root=root_dir, predefined_class_to_idx=class_to_idx)
        
        # Expose mapping for main script
        self.class_to_idx = self.image_folder.class_to_idx
        self.classes = self.image_folder.classes
        
        # Log status
        mode = "Train" if is_train else "Val"
        print(f"[{mode}] Loaded {len(self.classes)} classes (Filtered 'others').")
        
        # Geometry setup
        self.v_steps = grid_height
        self.u_steps = 2 * grid_height
        self.img_size = img_size
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps
        self.device = torch.device(device)
        
        # Pre-compute angles
        v_step_size = 180.0 / self.v_steps
        v_centers_deg = torch.linspace(90 - v_step_size / 2, -90 + v_step_size / 2, self.v_steps)
        u_centers_deg = torch.linspace(-180, 180, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers_deg, u_centers_deg, indexing='ij')
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)
        
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.samples[idx]
        try:
            with open(path, 'rb') as f:
                pano_pil = Image.open(f).convert('RGB')
                pano_np = np.array(pano_pil)
            
            patches_np_list = util.odi_processing.extract_patches_from_pano(
                pano_np=pano_np, h_num=self.u_steps, v_num=self.v_steps,
                h_fov_deg=self.h_fov, v_fov_deg=self.v_fov, out_size=self.img_size
            )
            
            if self.transform:
                patches = torch.stack([self.transform(Image.fromarray(p)) for p in patches_np_list])
            else:
                patches = torch.stack([transforms.ToTensor()(p) for p in patches_np_list])
                
        except Exception as e:
            print(f"Error loading {path}: {e}")
            N = self.u_steps * self.v_steps
            patches = torch.zeros(N, 3, self.img_size, self.img_size)
            
        patches = patches.to(self.device, non_blocking=True)
        angles = self.angle_centers.to(self.device, non_blocking=True)
        label = torch.tensor(label, device=self.device, dtype=torch.long)

        return patches, angles, label

def build_classification_dataset(is_train, args, class_to_idx=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    v_steps = args.grid_height
    dynamic_img_size = 512 // v_steps

    if is_train:
        t = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if hasattr(args, 'reprob') and args.reprob > 0:
            t.append(transforms.RandomErasing(p=args.reprob, value='random'))
        transform = transforms.Compose(t)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    dataset = PanoClassificationDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=dynamic_img_size,
        transform=transform,
        is_train=is_train,
        device=args.device,
        class_to_idx=class_to_idx # Pass mapping
    )
    return dataset