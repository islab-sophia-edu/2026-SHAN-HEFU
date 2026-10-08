import os
import glob
import random
import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ERPRotator:
    """
    Full-ERP yaw/pitch/roll augmentation.

    RGB uses bicubic interpolation. Semantic masks use nearest interpolation to
    preserve class ids. This is ERP-level augmentation only; tokenization remains
    pure grid patchify.
    """

    def __init__(self, h: int, w: int, device: str = "cpu"):
        self.h = int(h)
        self.w = int(w)
        self.device = torch.device(device)

        v, u = torch.meshgrid(
            torch.linspace(1, -1, self.h, device=self.device),
            torch.linspace(-1, 1, self.w, device=self.device),
            indexing="ij",
        )
        theta = u * np.pi
        phi = v * np.pi / 2.0
        self.xyz = torch.stack(
            [
                torch.cos(phi) * torch.cos(theta),
                -torch.cos(phi) * torch.sin(theta),
                torch.sin(phi),
            ],
            dim=-1,
        ).view(-1, 3)

    def rotate_tensor(
        self,
        img_tensor: torch.Tensor,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        mode: str = "bicubic",
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
                [np.cos(p_), 0.0, np.sin(p_)],
                [0.0,        1.0, 0.0],
                [-np.sin(p_), 0.0, np.cos(p_)],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [
                [1.0, 0.0,         0.0],
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
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )
        return rotated.squeeze(0).to(orig_device)


class Stanford2D3DGridDataset(Dataset):
    """
    Stanford 2D-3D-S semantic segmentation dataset for grid-token ablation.

    Tokenization:
        Pure ERP grid patchify, MAE-style row-major patch sequence.
        No ODI, no tangent-plane projection, no FoV sampling, no overlap.

    Returns before DataLoader batching:
        GCTT-grid mode:
            views, angles, gauge_angles, masks
        PB/center-only mode (--no_gctt):
            views, angles, masks

    `views` can be RGB-only (C=3) or RGB + ERP unit direction channels (C=6).
    The XYZ channels are fixed ERP grid directions, not tangent-plane rays.
    """

    def __init__(
        self,
        root_dir: str,
        grid_height: int = 16,
        img_size=None,
        is_train: bool = True,
        num_classes: int = 13,
        debug_limit: Optional[int] = None,
        args=None,
    ):
        self.root_dir = root_dir
        self.is_train = bool(is_train)
        self.num_classes = int(num_classes)
        self.args = args

        self.v_steps = int(grid_height)
        self.u_steps = 2 * int(grid_height)
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps

        self.mask_folder_name = getattr(args, "mask_folder_name", "anno") if args is not None else "anno"
        self.rgb_folder_name = getattr(args, "rgb_folder_name", "rgb") if args is not None else "rgb"

        if self.is_train:
            areas_str = getattr(args, "train_areas", "area_1,area_2,area_3,area_4,area_6") if args else "area_1,area_2,area_3,area_4,area_6"
        else:
            areas_str = getattr(args, "val_areas", "area_5a,area_5b") if args else "area_5a,area_5b"
        self.areas = [a.strip() for a in areas_str.split(",") if a.strip()]

        self.target_h = None
        self.target_w = None
        if args is not None and hasattr(args, "pano_h") and hasattr(args, "pano_w"):
            self.target_h = int(args.pano_h)
            self.target_w = int(args.pano_w)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.target_h = int(img_size[0]) * self.v_steps
            self.target_w = int(img_size[1]) * self.u_steps

        if self.target_h is not None and self.target_w is not None:
            if self.target_h % self.v_steps != 0 or self.target_w % self.u_steps != 0:
                raise ValueError(
                    f"pano_h/pano_w must be divisible by grid dimensions: "
                    f"{self.target_h}x{self.target_w} vs {self.v_steps}x{self.u_steps}."
                )
            self.patch_h = self.target_h // self.v_steps
            self.patch_w = self.target_w // self.u_steps
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.patch_h, self.patch_w = int(img_size[0]), int(img_size[1])
        elif isinstance(img_size, int):
            self.patch_h = self.patch_w = int(img_size)
        else:
            self.patch_h = self.patch_w = 128

        self.in_chans = int(getattr(args, "in_chans", 6)) if args is not None else 6
        self.use_xyz_channels = bool(getattr(args, "use_xyz_channels", True)) if args is not None else True
        if self.in_chans == 3:
            self.use_xyz_channels = False
        if self.use_xyz_channels and self.in_chans != 6:
            raise ValueError("XYZ channel mode expects --in_chans 6. Use --no_xyz_channels with --in_chans 3.")

        print(f"[{'Train' if self.is_train else 'Val'}] Loading Stanford2D3D GRID Segmentation")
        print(f"[{'Train' if self.is_train else 'Val'}] Root       : {self.root_dir}")
        print(f"[{'Train' if self.is_train else 'Val'}] Areas      : {self.areas}")
        print(f"[{'Train' if self.is_train else 'Val'}] RGB folder : {self.rgb_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Mask folder: {self.mask_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Patch Size : {self.patch_h}x{self.patch_w}")
        print(f"[{'Train' if self.is_train else 'Val'}] In channels: {self.in_chans} ({'RGB+ERP-XYZ' if self.use_xyz_channels else 'RGB'})")
        print(f"[{'Train' if self.is_train else 'Val'}] Tokenizer  : ERP grid patchify; no ODI / no tangent-plane")

        self.filenames = self._collect_pairs(debug_limit)
        print(f"[{'Train' if self.is_train else 'Val'}] Found {len(self.filenames)} RGB/mask pairs.")
        if len(self.filenames) == 0:
            raise ValueError(
                "Stanford dataset is empty. Check --data_path / --val_data_path, "
                "--train_areas / --val_areas, and the pano/rgb + pano/anno folder names."
            )

        v_centers = torch.linspace(90.0 - self.v_fov / 2.0, -90.0 + self.v_fov / 2.0, self.v_steps)
        u_centers = torch.linspace(-180.0, 180.0, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing="ij")
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).float()

        self.use_gctt = bool(getattr(args, "use_gctt", True)) if args is not None else True
        self.gctt_gauge_jitter_deg = 0.0
        self.gctt_local_gauge_jitter_deg = 0.0
        self.angle_jitter_deg = 0.0

        aug_device = getattr(args, "aug_device", "cuda") if args is not None else "cuda"
        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else "cpu")

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if bool(getattr(args, "use_color_jitter", True)) and self.is_train else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if bool(getattr(args, "use_blur", True)) and self.is_train else None
        )

        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", True)) and self.is_train
        self.pose_yaw_deg = float(getattr(args, "pose_yaw_deg", 360.0)) if args is not None else 360.0
        self.pose_pitch_deg = float(getattr(args, "pose_pitch_deg", 30.0)) if args is not None else 30.0
        self.pose_roll_deg = float(getattr(args, "pose_roll_deg", 30.0)) if args is not None else 30.0
        self.use_horizontal_roll = bool(getattr(args, "use_horizontal_roll", True)) and self.is_train
        self.hflip_prob = float(getattr(args, "hflip_prob", 0.0)) if self.is_train and args is not None else 0.0

        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._erp_xyz_cache = None
        self._erp_xyz_cache_size = None

    def _collect_pairs(self, debug_limit: Optional[int]):
        pairs = []
        examples_missing = []
        for area in self.areas:
            rgb_dir = os.path.join(self.root_dir, area, "pano", self.rgb_folder_name)
            mask_dir = os.path.join(self.root_dir, area, "pano", self.mask_folder_name)
            if not os.path.isdir(rgb_dir) or not os.path.isdir(mask_dir):
                print(f"[Skip] {area}: missing {rgb_dir} or {mask_dir}")
                continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
            rgb_files += sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
            rgb_files += sorted(glob.glob(os.path.join(rgb_dir, "*.jpeg")))
            mask_index = self._build_mask_index(mask_dir)

            area_pairs = 0
            for rgb_path in rgb_files:
                mask_path = self._find_mask_for_rgb(rgb_path, mask_index)
                if mask_path is not None:
                    pairs.append({"rgb": rgb_path, "mask": mask_path, "area": area})
                    area_pairs += 1
                elif len(examples_missing) < 8:
                    examples_missing.append(rgb_path)
            print(f"[Area] {area}: RGB={len(rgb_files)}, paired={area_pairs}")

        if examples_missing:
            print("Example RGB files without matched mask:", examples_missing[:8])

        if debug_limit:
            pairs = pairs[:debug_limit]
        return pairs

    @staticmethod
    def _norm_key(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        drop = {
            "rgb", "rgba", "color", "colour", "pano", "panorama",
            "anno", "annotation", "semantic", "semantics", "label", "labels",
            "mask", "masks", "seg", "segmentation", "class", "classes",
        }
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t and t not in drop]
        return "_".join(tokens)

    def _build_mask_index(self, mask_dir: str):
        mask_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
        index = {}
        for p in sorted(mask_files):
            index[self._norm_key(p)] = p
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            index[stem] = p
        return index

    def _find_mask_for_rgb(self, rgb_path: str, mask_index: dict):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        candidates = [
            base.replace("_rgb", f"_{self.mask_folder_name}"),
            base.replace("rgb", self.mask_folder_name),
            base + f"_{self.mask_folder_name}",
            base,
            self._norm_key(rgb_path),
        ]
        for cand in candidates:
            k = cand.lower()
            if k in mask_index:
                return mask_index[k]
        return None

    def _get_rotator(self, h: int, w: int):
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h = h
            self._rotator_w = w
        return self._rotator

    def _apply_full_pose3d(self, img_np: np.ndarray, mask_np: np.ndarray):
        if not self.use_full_pose3d:
            return img_np, mask_np

        yaw = random.uniform(0.0, self.pose_yaw_deg) if self.pose_yaw_deg > 0 else 0.0
        pitch = random.uniform(-self.pose_pitch_deg, self.pose_pitch_deg) if self.pose_pitch_deg > 0 else 0.0
        roll = random.uniform(-self.pose_roll_deg, self.pose_roll_deg) if self.pose_roll_deg > 0 else 0.0

        rgb_t = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        rotator = self._get_rotator(img_np.shape[0], img_np.shape[1])
        rgb_rot = rotator.rotate_tensor(rgb_t, yaw=yaw, pitch=pitch, roll=roll, mode="bicubic")
        mask_rot = rotator.rotate_tensor(mask_t, yaw=yaw, pitch=pitch, roll=roll, mode="nearest")

        img_np = (rgb_rot.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        mask_np = mask_rot.squeeze(0).round().cpu().numpy().astype(mask_np.dtype)
        return img_np, mask_np

    def __len__(self):
        return len(self.filenames)

    def _make_angles_and_gauges(self):
        angles = self.angle_centers.clone().float()
        gauge_angles = torch.zeros(angles.shape[0], 1, dtype=torch.float32)
        return angles, gauge_angles

    def _patchify_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [C, H, W]
        c, h, w = tensor.shape
        expected_h = self.v_steps * self.patch_h
        expected_w = self.u_steps * self.patch_w
        if h != expected_h or w != expected_w:
            raise ValueError(f"Unexpected tensor size {h}x{w}; expected {expected_h}x{expected_w}.")
        patches = tensor.view(c, self.v_steps, self.patch_h, self.u_steps, self.patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()
        return patches.view(self.v_steps * self.u_steps, c, self.patch_h, self.patch_w)

    def _patchify_mask_grid(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        h, w = mask_tensor.shape
        expected_h = self.v_steps * self.patch_h
        expected_w = self.u_steps * self.patch_w
        if h != expected_h or w != expected_w:
            raise ValueError(f"Unexpected mask size {h}x{w}; expected {expected_h}x{expected_w}.")
        patches = mask_tensor.view(self.v_steps, self.patch_h, self.u_steps, self.patch_w)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.view(self.v_steps * self.u_steps, self.patch_h, self.patch_w)

    def _erp_xyz(self, h: int, w: int) -> torch.Tensor:
        cache_key = (h, w)
        if self._erp_xyz_cache is not None and self._erp_xyz_cache_size == cache_key:
            return self._erp_xyz_cache

        v, u = torch.meshgrid(
            torch.linspace(1.0, -1.0, h),
            torch.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        theta = u * np.pi
        phi = v * np.pi / 2.0
        xyz = torch.stack(
            [
                torch.cos(phi) * torch.cos(theta),
                -torch.cos(phi) * torch.sin(theta),
                torch.sin(phi),
            ],
            dim=0,
        ).float()
        self._erp_xyz_cache = xyz
        self._erp_xyz_cache_size = cache_key
        return xyz

    def __getitem__(self, idx: int):
        data = self.filenames[idx]
        try:
            img_pil = Image.open(data["rgb"]).convert("RGB")
            mask_pil = Image.open(data["mask"]).convert("RGB")

            if self.target_h is not None and self.target_w is not None:
                if img_pil.size != (self.target_w, self.target_h):
                    img_pil = img_pil.resize((self.target_w, self.target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((self.target_w, self.target_h), Image.NEAREST)

            if self.color_jitter is not None and random.random() < 0.8:
                img_pil = self.color_jitter(img_pil)
            if self.blur is not None and random.random() < 0.1:
                img_pil = self.blur(img_pil)

            img_np = np.array(img_pil)
            mask_raw = np.array(mask_pil)
            mask_np = mask_raw[:, :, 0].astype(np.int64) if mask_raw.ndim == 3 else mask_raw.astype(np.int64)

            img_np, mask_np = self._apply_full_pose3d(img_np, mask_np)

            if self.use_horizontal_roll:
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1).copy()
                mask_np = np.roll(mask_np, roll_idx, axis=1).copy()

            if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                img_np = np.flip(img_np, axis=1).copy()
                mask_np = np.flip(mask_np, axis=1).copy()

            angles, gauge_angles = self._make_angles_and_gauges()

            rgb_tensor = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_np.astype(np.int64)).long()

            rgb_patches = self._patchify_grid(rgb_tensor)
            rgb_patches = torch.stack([self.normalize(p) for p in rgb_patches], dim=0)
            mask_patches = self._patchify_mask_grid(mask_tensor)

            if self.use_xyz_channels:
                xyz_patches = self._patchify_grid(self._erp_xyz(rgb_tensor.shape[1], rgb_tensor.shape[2]))
                views = torch.cat([rgb_patches, xyz_patches], dim=1)
            else:
                views = rgb_patches

        except Exception as e:
            print(f"[Error] Loading {data.get('rgb', 'unknown')}: {e}")
            n = self.u_steps * self.v_steps
            views = torch.zeros(n, self.in_chans, self.patch_h, self.patch_w)
            angles = self.angle_centers.clone()
            gauge_angles = torch.zeros(n, 1)
            mask_patches = torch.zeros(n, self.patch_h, self.patch_w).long()

        if self.use_gctt:
            return views, angles, gauge_angles, mask_patches
        return views, angles, mask_patches


def build_segmentation_dataset(is_train, args):
    return Stanford2D3DGridDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=getattr(args, "input_size", None),
        is_train=is_train,
        num_classes=args.nb_classes,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
