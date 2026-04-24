import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

from odi_utils import get_perspective_patch

class PanoramicClassifierDataset(Dataset):
    """
    A Dataset for panoramic image classification.
    It treats each generated perspective view from a panorama as a distinct sample.
    This acts as a powerful form of data augmentation.
    """
    def __init__(self, root_dir, transform=None, patch_size=224, fov=60, u_steps=8, v_steps=4):
        
        # Use ImageFolder to easily get image paths and their labels
        self.image_folder = ImageFolder(root=root_dir)
        self.transform = transform
        
        # View generation parameters
        self.patch_size = patch_size
        self.fov = fov
        self.u_steps = u_steps
        self.v_steps = v_steps
        self.views_per_image = u_steps * v_steps
        
        # Pre-calculate the angle grid for all views
        u_angles = np.linspace(-180, 180, u_steps, endpoint=False)
        v_angles = np.linspace(-75, 75, v_steps, endpoint=True)
        self.angle_grid = []
        for v_deg in v_angles:
            for u_deg in u_angles:
                self.angle_grid.append({'u': u_deg, 'v': v_deg})

    def __len__(self):
        # The total dataset size is the number of panoramas * views per panorama
        return len(self.image_folder) * self.views_per_image

    def __getitem__(self, idx):
        # Determine which panorama and which view to generate
        original_image_idx = idx // self.views_per_image
        view_idx = idx % self.views_per_image
        
        # Get the original panorama path and its label
        original_image_path, label = self.image_folder.samples[original_image_idx]
        
        # Get the specific angles for the desired view
        angles = self.angle_grid[view_idx]
        u_deg, v_deg = angles['u'], angles['v']
        
        # Open the original panoramic image
        try:
            pano_image_pil = Image.open(original_image_path).convert('RGB')
            pano_image_np = np.array(pano_image_pil)
        except Exception as e:
            print(f"Error loading image {original_image_path}: {e}")
            # On error, load the next sample
            return self.__getitem__((idx + 1) % len(self))

        # Generate ONLY the required single perspective view
        view_patch = get_perspective_patch(
            pano_image=pano_image_np,
            v_fov=self.fov,
            h_fov=self.fov,
            u_deg=u_deg,
            v_deg=v_deg,
            output_height=self.patch_size,
            output_width=self.patch_size
        )

        # Apply transformations (e.g., ToTensor, Normalize)
        # Note: The fine-tune script has its own complex transforms,
        # we will pass them in during initialization.
        if self.transform:
            view_patch = self.transform(Image.fromarray(view_patch))

        return view_patch, label