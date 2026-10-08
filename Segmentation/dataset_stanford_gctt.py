import os
import glob
import random
import re
from typing import List, Optional, Tuple

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
    keep class ids intact. This is enabled only for is_train=True.
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
                [np.sin(y_), np.cos(y_), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        Ry = torch.tensor(
            [
                [np.cos(p_), 0.0, np.sin(p_)],
                [0.0, 1.0, 0.0],
                [-np.sin(p_), 0.0, np.cos(p_)],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(r_), -np.sin(r_)],
                [0.0, np.sin(r_), np.cos(r_)],
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


class Stanford2D3DGCTTDataset(Dataset):
    """
    Stanford 2D-3D-S semantic segmentation dataset for the GCTT segmentation engine.

    Directory layout expected under root_dir:
        area_x/pano/rgb/*.png
        area_x/pano/anno/*.png       # default mask folder

    Returns, before DataLoader batching:
        views        : [N, C, patch_h, patch_w]
                       C=6 by default: normalized RGB + unit-ray XYZ channels.
        angles       : [N, 2], degrees, ordered as [lon/theta, lat/phi].
        gauge_angles : [N, 1], degrees, local tangent-frame in-plane gauge psi_i.
        masks        : [N, patch_h, patch_w], int64 labels.

    The RGB view, optional XYZ/ray channels, and mask target are produced with the
    same tangent-frame gauge, so they stay aligned under GCTT gauge jitter.
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

        # Resolve target panorama and patch sizes.
        self.target_h = None
        self.target_w = None
        if args is not None and hasattr(args, "pano_h") and hasattr(args, "pano_w"):
            self.target_h = int(args.pano_h)
            self.target_w = int(args.pano_w)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.target_h = int(img_size[0]) * self.v_steps
            self.target_w = int(img_size[1]) * self.u_steps

        if self.target_h is not None and self.target_w is not None:
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

        print(f"[{'Train' if self.is_train else 'Val'}] Loading Stanford2D3D GCTT Segmentation")
        print(f"[{'Train' if self.is_train else 'Val'}] Root       : {self.root_dir}")
        print(f"[{'Train' if self.is_train else 'Val'}] Areas      : {self.areas}")
        print(f"[{'Train' if self.is_train else 'Val'}] RGB folder : {self.rgb_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Mask folder: {self.mask_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Patch Size : {self.patch_h}x{self.patch_w}")
        print(f"[{'Train' if self.is_train else 'Val'}] In channels: {self.in_chans} ({'RGB+XYZ' if self.use_xyz_channels else 'RGB'})")

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
        self.gctt_gauge_jitter_deg = float(getattr(args, "gctt_gauge_jitter_deg", 30.0)) if self.is_train else 0.0
        self.gctt_local_gauge_jitter_deg = float(getattr(args, "gctt_local_gauge_jitter_deg", 0.0)) if self.is_train else 0.0
        self.angle_jitter_deg = float(getattr(args, "angle_jitter_deg", 0.0)) if self.is_train else 0.0

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

        u_lin = torch.linspace(-(self.patch_w - 1) / 2.0, (self.patch_w - 1) / 2.0, self.patch_w)
        v_lin = torch.linspace((self.patch_h - 1) / 2.0, -(self.patch_h - 1) / 2.0, self.patch_h)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing="xy")
        self._pixel_u = uu.reshape(-1)
        self._pixel_v = vv.reshape(-1)

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
        angles = self.angle_centers.clone()

        if self.angle_jitter_deg > 0.0:
            jitter = torch.empty_like(angles).uniform_(-self.angle_jitter_deg, self.angle_jitter_deg)
            angles = angles + jitter
            lat_min = -90.0 + self.v_fov / 2.0
            lat_max = 90.0 - self.v_fov / 2.0
            angles[:, 1].clamp_(lat_min, lat_max)

        if self.use_gctt and self.is_train and self.gctt_gauge_jitter_deg > 0.0:
            global_gauge = random.uniform(-self.gctt_gauge_jitter_deg, self.gctt_gauge_jitter_deg)
        else:
            global_gauge = 0.0

        if self.use_gctt and self.is_train and self.gctt_local_gauge_jitter_deg > 0.0:
            local = torch.empty(angles.shape[0], 1).uniform_(
                -self.gctt_local_gauge_jitter_deg,
                self.gctt_local_gauge_jitter_deg,
            )
        else:
            local = torch.zeros(angles.shape[0], 1)

        gauge_angles = torch.full((angles.shape[0], 1), float(global_gauge)) + local
        if not self.use_gctt:
            gauge_angles.zero_()
        return angles.float(), gauge_angles.float()

    def _extract_patches_gpu(
        self,
        pano_tensor: torch.Tensor,
        angles_deg: torch.Tensor,
        h_fov_rad: float,
        v_fov_rad: float,
        gauge_angles_deg: Optional[torch.Tensor] = None,
        mode: str = "bicubic",
        return_rays: bool = False,
    ):
        device = self.aug_device
        pano_tensor = pano_tensor.to(device, non_blocking=True)
        angles_t = angles_deg.to(device=device, dtype=torch.float32)

        N = angles_t.shape[0]
        theta = torch.deg2rad(angles_t[:, 0])
        phi = torch.deg2rad(angles_t[:, 1])

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        nc = torch.stack([cos_phi * cos_theta, -cos_phi * sin_theta, sin_phi], dim=1)
        xn = torch.stack([-sin_theta, -cos_theta, torch.zeros_like(theta)], dim=1)
        yn = torch.stack([-sin_phi * cos_theta, sin_phi * sin_theta, cos_phi], dim=1)

        if gauge_angles_deg is None:
            gauge = torch.zeros(N, device=device, dtype=torch.float32)
        else:
            gauge = gauge_angles_deg.to(device=device, dtype=torch.float32).view(N)
        psi = torch.deg2rad(gauge)
        c = torch.cos(psi).unsqueeze(1)
        s = torch.sin(psi).unsqueeze(1)
        xg = c * xn + s * yn
        yg = -s * xn + c * yn

        fx = (self.patch_w / 2.0) / torch.tan(torch.tensor(h_fov_rad / 2.0, device=device))
        fy = (self.patch_h / 2.0) / torch.tan(torch.tensor(v_fov_rad / 2.0, device=device))

        uu = self._pixel_u.to(device)
        vv = self._pixel_v.to(device)
        pts = (
            (uu.view(1, -1, 1) / fx) * xg.unsqueeze(1)
            + (vv.view(1, -1, 1) / fy) * yg.unsqueeze(1)
            + nc.unsqueeze(1)
        )

        dirs = F.normalize(pts, dim=-1)
        px, py, pz = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        theta_erp = torch.atan2(-py, px)
        phi_erp = torch.asin(torch.clamp(pz, -1.0, 1.0))

        grid_x = theta_erp / torch.pi
        grid_y = -phi_erp / (torch.pi / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, self.patch_h, self.patch_w, 2)

        pano_batch = pano_tensor.unsqueeze(0).expand(N, -1, -1, -1)
        patches = F.grid_sample(
            pano_batch,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )

        if return_rays:
            rays = dirs.view(N, self.patch_h, self.patch_w, 3).permute(0, 3, 1, 2)
            return patches, rays
        return patches

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

            # Horizontal reflection is disabled by default. It is not part of the
            # PanoMAE pretraining augmentation and can invert tangent-frame handedness.
            if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                img_np = np.flip(img_np, axis=1).copy()
                mask_np = np.flip(mask_np, axis=1).copy()

            angles, gauge_angles = self._make_angles_and_gauges()

            rgb_tensor = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

            rgb_patches, ray_patches = self._extract_patches_gpu(
                rgb_tensor,
                angles,
                np.radians(self.h_fov),
                np.radians(self.v_fov),
                gauge_angles_deg=gauge_angles if self.use_gctt else None,
                mode="bicubic",
                return_rays=True,
            )
            mask_patches = self._extract_patches_gpu(
                mask_tensor,
                angles,
                np.radians(self.h_fov),
                np.radians(self.v_fov),
                gauge_angles_deg=gauge_angles if self.use_gctt else None,
                mode="nearest",
                return_rays=False,
            )

            rgb_patches = rgb_patches.detach().cpu()
            ray_patches = ray_patches.detach().cpu()
            mask_patches = mask_patches[:, 0].round().long().detach().cpu()

            rgb_patches = torch.stack([self.normalize(p) for p in rgb_patches], dim=0)
            if self.use_xyz_channels:
                views = torch.cat([rgb_patches, ray_patches], dim=1)
            else:
                views = rgb_patches

        except Exception as e:
            print(f"[Error] Loading {data.get('rgb', 'unknown')}: {e}")
            n = self.u_steps * self.v_steps
            return (
                torch.zeros(n, self.in_chans, self.patch_h, self.patch_w),
                self.angle_centers.clone(),
                torch.zeros(n, 1),
                torch.zeros(n, self.patch_h, self.patch_w).long(),
            )

        return views, angles, gauge_angles, mask_patches


def build_segmentation_dataset(is_train, args):
    return Stanford2D3DGCTTDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=getattr(args, "input_size", None),
        is_train=is_train,
        num_classes=args.nb_classes,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
