import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class StrictImageFolder(ImageFolder):
    """
    ImageFolder with two project-specific rules:
      1. Remove the 'others' class automatically.
      2. Reuse the training class_to_idx mapping for validation/test.
    """

    def __init__(
        self,
        root: str,
        predefined_class_to_idx: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self.predefined_class_to_idx = predefined_class_to_idx
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        if "others" in classes:
            classes.remove("others")

        if self.predefined_class_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx

        valid_classes = [c for c in classes if c in self.predefined_class_to_idx]
        class_to_idx = {c: self.predefined_class_to_idx[c] for c in valid_classes}
        return valid_classes, class_to_idx


class ERPRotator:
    """
    Optional SO(3) rotation on the full ERP image.

    This is only a full-panorama augmentation. It is not ODI tokenization and it
    is not tangent-plane patch extraction. Grid tokens are still created by plain
    rectangular ERP patchify after this optional augmentation.

    Coordinate convention:
        p(theta, phi) = [cos(phi) cos(theta), -cos(phi) sin(theta), sin(phi)].
    """

    def __init__(self, h: int, w: int, device: str = "cpu"):
        self.h = h
        self.w = w
        self.device = torch.device(device)

        v, u = torch.meshgrid(
            torch.linspace(1, -1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing="ij",
        )
        self.theta = u * np.pi
        self.phi = v * np.pi / 2.0
        self.xyz = torch.stack(
            [
                torch.cos(self.phi) * torch.cos(self.theta),
                -torch.cos(self.phi) * torch.sin(self.theta),
                torch.sin(self.phi),
            ],
            dim=-1,
        ).view(-1, 3)

    def rotate_tensor(
        self,
        img_tensor: torch.Tensor,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> torch.Tensor:
        if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
            return img_tensor

        orig_device = img_tensor.device
        img_4d = img_tensor.to(self.device).unsqueeze(0)

        y_, p_, r_ = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor(
            [
                [np.cos(y_), -np.sin(y_), 0.0],
                [np.sin(y_),  np.cos(y_), 0.0],
                [0.0,         0.0,        1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        Ry = torch.tensor(
            [
                [ np.cos(p_), 0.0, np.sin(p_)],
                [ 0.0,        1.0, 0.0       ],
                [-np.sin(p_), 0.0, np.cos(p_)],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [
                [1.0, 0.0,         0.0        ],
                [0.0, np.cos(r_), -np.sin(r_)],
                [0.0, np.sin(r_),  np.cos(r_)],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        R = Rz @ Ry @ Rx

        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack(
            [
                torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi,
                -(torch.asin(torch.clamp(xyz_rot[:, 2], -1.0, 1.0)) / (np.pi / 2.0)),
            ],
            dim=-1,
        ).view(1, self.h, self.w, 2)

        rotated = F.grid_sample(
            img_4d,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        return rotated.squeeze(0).to(orig_device)


class PanoClassificationGridDataset(Dataset):
    """
    Pure ERP-grid classification dataset for the GCTT/PB tokenization ablation.

    Tokenization:
        - No ODI.
        - No tangent-plane projection.
        - No FoV sampling.
        - The resized ERP panorama is split into a fixed row-major grid, exactly
          like MAE patchify/unfold.

    Returns:
        GCTT-compatible mode:
            views, angles, gauge_angles, label
        PB / non-GCTT mode:
            views, angles, label

    views:
        [N, 3, S, S], normalized ERP grid patches.
    angles:
        [N, 2], patch-center ERP spherical coordinates in degrees, ordered as
        [theta/lon, phi/lat].
    gauge_angles:
        [N, 1], fixed zeros in grid mode.

    Why zero gauge:
        After tangent tokenization is removed, random gauge jitter would change
        only the positional/oriented-frame state while the rectangular ERP patch
        content is not rotated accordingly. That would create a confounded
        ablation. Therefore grid-GCTT keeps psi_i = 0.
    """

    def __init__(
        self,
        root_dir: str,
        grid_height: int = 4,
        img_size: int = 128,
        pano_h: int = 512,
        pano_w: int = 1024,
        is_train: bool = True,
        class_to_idx: Optional[Dict[str, int]] = None,
        # Image augmentation on the full ERP image.
        use_color_jitter: bool = True,
        use_blur: bool = True,
        use_full_pose3d: bool = False,
        use_horizontal_roll: bool = False,
        # Compatibility flags kept so old command lines do not break.
        multiscale_sampling: bool = False,
        angle_jitter_deg: float = 0.0,
        use_gctt: bool = True,
        gctt_gauge_jitter_deg: float = 0.0,
        gctt_local_gauge_jitter_deg: float = 0.0,
        # Runtime.
        aug_device: str = "cpu",
        return_device: str = "cpu",
        reprob: float = 0.0,
    ):
        del multiscale_sampling, angle_jitter_deg, gctt_gauge_jitter_deg, gctt_local_gauge_jitter_deg

        self.root_dir = root_dir
        self.image_folder = StrictImageFolder(root=root_dir, predefined_class_to_idx=class_to_idx)
        self.class_to_idx = self.image_folder.class_to_idx
        self.classes = self.image_folder.classes
        self.is_train = bool(is_train)

        mode = "Train" if is_train else "Val"
        print(f"[{mode}] Loaded {len(self.classes)} classes (filtered 'others').")

        self.pano_h = int(pano_h)
        self.pano_w = int(pano_w)
        self.grid_height = int(grid_height)
        self.v_num = self.grid_height
        self.u_num = 2 * self.grid_height

        if self.pano_h % self.v_num != 0:
            raise ValueError(
                f"pano_h={self.pano_h} must be divisible by grid_height={self.v_num}."
            )
        if self.pano_w % self.u_num != 0:
            raise ValueError(
                f"pano_w={self.pano_w} must be divisible by 2*grid_height={self.u_num}."
            )

        patch_h = self.pano_h // self.v_num
        patch_w = self.pano_w // self.u_num
        if patch_h != patch_w:
            raise ValueError(
                f"This classifier embeds square MAE-style patches, but the grid patch is "
                f"{patch_h}x{patch_w}. Use a 2:1 panorama with u_num=2*grid_height."
            )
        if img_size is not None and int(img_size) != patch_h:
            raise ValueError(
                f"img_size={img_size} is inconsistent with ERP grid patch size "
                f"{patch_h}x{patch_w}. Use img_size={patch_h} or adjust pano/grid."
            )
        self.patch_size = patch_h

        self.use_gctt = bool(use_gctt)
        self.use_full_pose3d = bool(use_full_pose3d and is_train)
        self.use_horizontal_roll = bool(use_horizontal_roll and is_train)

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if (use_color_jitter and is_train)
            else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if (use_blur and is_train)
            else None
        )
        self.reprob = float(reprob) if is_train else 0.0

        self.aug_device = torch.device(
            aug_device if (str(aug_device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.return_device = torch.device(return_device)

        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.base_angles_deg = torch.tensor(self._generate_grid_angles_deg(), dtype=torch.float32)
        self.zero_gauge = torch.zeros(self.v_num * self.u_num, 1, dtype=torch.float32)

        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

    def _generate_grid_angles_deg(self) -> np.ndarray:
        """Return row-major patch-center coordinates [theta, phi] in degrees."""
        rows = np.arange(self.v_num, dtype=np.float32)
        cols = np.arange(self.u_num, dtype=np.float32)

        theta = -180.0 + (cols + 0.5) * (360.0 / self.u_num)
        phi = 90.0 - (rows + 0.5) * (180.0 / self.v_num)
        grid_phi, grid_theta = np.meshgrid(phi, theta, indexing="ij")
        return np.stack([grid_theta.reshape(-1), grid_phi.reshape(-1)], axis=1)

    def _get_rotator(self, h: int, w: int) -> ERPRotator:
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h = h
            self._rotator_w = w
        return self._rotator

    def _load_panorama(self, idx: int) -> torch.Tensor:
        path, _ = self.image_folder.samples[idx]
        with open(path, "rb") as f:
            pano_pil = Image.open(f).convert("RGB")

        if pano_pil.size != (self.pano_w, self.pano_h):
            pano_pil = pano_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)

        if self.color_jitter is not None and random.random() < 0.8:
            pano_pil = self.color_jitter(pano_pil)

        if self.blur is not None and random.random() < 0.1:
            pano_pil = self.blur(pano_pil)

        if self.use_horizontal_roll:
            shift = np.random.randint(0, self.pano_w)
            pano_pil = Image.fromarray(np.roll(np.asarray(pano_pil), shift, axis=1))

        pano_tensor = transforms.ToTensor()(pano_pil).to(self.aug_device)

        if self.use_full_pose3d:
            yaw_shift = np.random.uniform(0.0, 360.0)
            pitch_shift = np.random.uniform(-30.0, 30.0)
            roll_shift = np.random.uniform(-30.0, 30.0)
            rotator = self._get_rotator(self.pano_h, self.pano_w)
            pano_tensor = rotator.rotate_tensor(
                pano_tensor,
                yaw=yaw_shift,
                pitch=pitch_shift,
                roll=roll_shift,
            )

        return pano_tensor

    def _patchify_grid(self, pano_tensor: torch.Tensor) -> torch.Tensor:
        """Split [3,H,W] ERP into row-major [N,3,P,P] grid patches."""
        C, H, W = pano_tensor.shape
        P = self.patch_size
        if H != self.pano_h or W != self.pano_w:
            raise ValueError(f"Unexpected tensor size {H}x{W}; expected {self.pano_h}x{self.pano_w}.")

        patches = (
            pano_tensor
            .view(C, self.v_num, P, self.u_num, P)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
            .view(self.v_num * self.u_num, C, P, P)
        )
        return patches

    @staticmethod
    def _random_erasing_batch(patches: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0.0:
            return patches
        eraser = transforms.RandomErasing(p=p, value="random")
        return torch.stack([eraser(patch) for patch in patches], dim=0)

    def __len__(self) -> int:
        return len(self.image_folder)

    def __getitem__(self, idx: int):
        path, label = self.image_folder.samples[idx]
        N = self.v_num * self.u_num

        try:
            pano_tensor = self._load_panorama(idx)
            patches = self._patchify_grid(pano_tensor)

            mean = self._norm_mean.to(patches.device)
            std = self._norm_std.to(patches.device)
            patches = (patches - mean) / std
            patches = self._random_erasing_batch(patches, self.reprob)

            angles = self.base_angles_deg.to(patches.device).clone()
            gauge_angles = self.zero_gauge.to(patches.device).clone()

        except Exception as e:
            print(f"Error loading {path}: {e}")
            patches = torch.zeros(N, 3, self.patch_size, self.patch_size)
            angles = torch.zeros(N, 2)
            gauge_angles = torch.zeros(N, 1)

        label = torch.tensor(label, dtype=torch.long, device=patches.device)

        if self.return_device.type == "cpu":
            patches = patches.detach().cpu()
            angles = angles.detach().cpu()
            gauge_angles = gauge_angles.detach().cpu()
            label = label.detach().cpu()
            if self.use_gctt:
                return patches, angles, gauge_angles, label
            return patches, angles, label

        patches = patches.to(self.return_device, non_blocking=True)
        angles = angles.to(self.return_device, non_blocking=True)
        label = label.to(self.return_device, non_blocking=True)
        if self.use_gctt:
            gauge_angles = gauge_angles.to(self.return_device, non_blocking=True)
            return patches, angles, gauge_angles, label
        return patches, angles, label


def build_classification_dataset(is_train, args, class_to_idx=None):
    dynamic_img_size = args.pano_h // args.grid_height

    dataset = PanoClassificationGridDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=dynamic_img_size,
        pano_h=args.pano_h,
        pano_w=args.pano_w,
        is_train=is_train,
        class_to_idx=class_to_idx,
        use_color_jitter=getattr(args, "use_color_jitter", True),
        use_blur=getattr(args, "use_blur", True),
        use_full_pose3d=getattr(args, "use_full_pose3d", False),
        use_horizontal_roll=getattr(args, "use_horizontal_roll", False),
        multiscale_sampling=getattr(args, "multiscale_sampling", False),
        angle_jitter_deg=getattr(args, "angle_jitter_deg", 0.0),
        use_gctt=getattr(args, "use_gctt", True),
        gctt_gauge_jitter_deg=0.0,
        gctt_local_gauge_jitter_deg=0.0,
        aug_device=getattr(args, "aug_device", "cpu"),
        return_device="cpu",
        reprob=getattr(args, "reprob", 0.0),
    )
    return dataset
