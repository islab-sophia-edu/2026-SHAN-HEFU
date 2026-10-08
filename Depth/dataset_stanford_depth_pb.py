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
    """PB-style full-ERP yaw/pitch/roll augmentation."""
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

    def rotate_tensor(self, img_tensor: torch.Tensor, yaw=0.0, pitch=0.0, roll=0.0, mode="bicubic"):
        if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
            return img_tensor
        orig_device = img_tensor.device
        img_4d = img_tensor.to(self.device).unsqueeze(0)
        y_, p_, r_ = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor(
            [[np.cos(y_), -np.sin(y_), 0.0], [np.sin(y_), np.cos(y_), 0.0], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        Ry = torch.tensor(
            [[np.cos(p_), 0.0, np.sin(p_)], [0.0, 1.0, 0.0], [-np.sin(p_), 0.0, np.cos(p_)]],
            device=self.device,
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, np.cos(r_), -np.sin(r_)], [0.0, np.sin(r_), np.cos(r_)]],
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
        rotated = F.grid_sample(img_4d, grid, mode=mode, padding_mode="border", align_corners=True)
        return rotated.squeeze(0).to(orig_device)


class Stanford2D3DDepthPBDataset(Dataset):
    """
    PB-compatible Stanford2D3D depth dataset.

    Returns before DataLoader batching:
        views      : [N, 3, patch_h, patch_w]
        angles     : [N, 2], [lon, lat] degrees
        depths     : [N, 1, patch_h, patch_w], normalized to [0, 1]
        valid_mask : [N, 1, patch_h, patch_w]

    No gauge_angles are generated or returned.
    """
    def __init__(self, root_dir: str, grid_height: int = 16, img_size=None, is_train=True, debug_limit: Optional[int] = None, args=None):
        self.root_dir = root_dir
        self.is_train = bool(is_train)
        self.args = args
        self.v_steps = int(grid_height)
        self.u_steps = 2 * int(grid_height)
        self.base_v_fov = 180.0 / self.v_steps
        self.base_h_fov = 360.0 / self.u_steps

        self.rgb_folder_name = getattr(args, "rgb_folder_name", "rgb") if args is not None else "rgb"
        self.depth_folder_name = getattr(args, "depth_folder_name", "depth") if args is not None else "depth"
        self.max_depth = float(getattr(args, "max_depth", 100.0)) if args is not None else 100.0
        self.depth_scale = float(getattr(args, "depth_scale", 512.0)) if args is not None else 512.0
        self.min_depth = float(getattr(args, "min_depth", 0.05)) if args is not None else 0.05
        self.depth_valid_min = float(getattr(args, "depth_valid_min", self.min_depth)) if args is not None else self.min_depth
        self.invalid_depth_raw = int(getattr(args, "invalid_depth_raw", 65535)) if args is not None else 65535

        areas_str = getattr(args, "train_areas" if self.is_train else "val_areas", None) if args else None
        if areas_str is None:
            areas_str = "area_1,area_2,area_3,area_4,area_6" if self.is_train else "area_5a,area_5b"
        self.areas = [a.strip() for a in areas_str.split(",") if a.strip()]

        self.target_h = int(getattr(args, "pano_h", 2048)) if args is not None else None
        self.target_w = int(getattr(args, "pano_w", 4096)) if args is not None else None
        if self.target_h is not None and self.target_w is not None:
            self.patch_h = self.target_h // self.v_steps
            self.patch_w = self.target_w // self.u_steps
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.patch_h, self.patch_w = int(img_size[0]), int(img_size[1])
        elif isinstance(img_size, int):
            self.patch_h = self.patch_w = int(img_size)
        else:
            self.patch_h = self.patch_w = 128

        self.in_chans = int(getattr(args, "in_chans", 3)) if args is not None else 3
        if self.in_chans != 3:
            raise ValueError("PB depth ablation is RGB-only. Use --in_chans 3.")

        self.filenames = self._collect_pairs(debug_limit)
        print(f"[{'Train' if self.is_train else 'Val'}] Stanford2D3D PB Depth | root={self.root_dir} | areas={self.areas} | pairs={len(self.filenames)}")
        if len(self.filenames) == 0:
            raise ValueError("Stanford depth dataset is empty. Check root, area split, and pano/rgb + pano/depth folders.")

        v_centers = torch.linspace(90.0 - self.base_v_fov / 2.0, -90.0 + self.base_v_fov / 2.0, self.v_steps)
        u_centers = torch.linspace(-180.0, 180.0, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing="ij")
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).float()

        self.multiscale_sampling = bool(getattr(args, "multiscale_sampling", False)) and self.is_train
        self.angle_jitter_deg = float(getattr(args, "angle_jitter_deg", 5.0)) if self.is_train else 0.0
        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", True)) and self.is_train
        self.pose_yaw_deg = float(getattr(args, "pose_yaw_deg", 360.0)) if args is not None else 360.0
        self.pose_pitch_deg = float(getattr(args, "pose_pitch_deg", 30.0)) if args is not None else 30.0
        self.pose_roll_deg = float(getattr(args, "pose_roll_deg", 30.0)) if args is not None else 30.0
        self.use_horizontal_roll = bool(getattr(args, "use_horizontal_roll", False)) and self.is_train
        self.hflip_prob = float(getattr(args, "hflip_prob", 0.0)) if self.is_train and args is not None else 0.0
        aug_device = getattr(args, "aug_device", "cuda") if args is not None else "cuda"
        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else "cpu")

        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) if bool(getattr(args, "use_color_jitter", True)) and self.is_train else None
        self.blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5)) if bool(getattr(args, "use_blur", True)) and self.is_train else None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

        u_lin = torch.linspace(-(self.patch_w - 1) / 2.0, (self.patch_w - 1) / 2.0, self.patch_w)
        v_lin = torch.linspace((self.patch_h - 1) / 2.0, -(self.patch_h - 1) / 2.0, self.patch_h)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing="xy")
        self._pixel_u = uu.reshape(-1)
        self._pixel_v = vv.reshape(-1)

    def _collect_pairs(self, debug_limit):
        pairs = []
        examples_missing = []
        for area in self.areas:
            rgb_dir = os.path.join(self.root_dir, area, "pano", self.rgb_folder_name)
            depth_dir = os.path.join(self.root_dir, area, "pano", self.depth_folder_name)
            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
                print(f"[Skip] {area}: missing {rgb_dir} or {depth_dir}")
                continue
            rgb_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
            depth_index = self._build_depth_index(depth_dir)
            area_pairs = 0
            for rgb_path in sorted(rgb_files):
                depth_path = self._find_depth_for_rgb(rgb_path, depth_index)
                if depth_path is not None:
                    pairs.append({"rgb": rgb_path, "depth": depth_path, "area": area})
                    area_pairs += 1
                elif len(examples_missing) < 8:
                    examples_missing.append(rgb_path)
            print(f"[Area] {area}: RGB={len(rgb_files)}, paired={area_pairs}")
        if examples_missing:
            print("Example RGB files without matched depth:", examples_missing[:8])
        if debug_limit:
            pairs = pairs[:debug_limit]
        return pairs

    @staticmethod
    def _norm_key(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        drop = {"rgb", "rgba", "color", "colour", "pano", "panorama", "depth", "depths", "z", "metric", "meter", "meters"}
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t and t not in drop]
        return "_".join(tokens)

    def _build_depth_index(self, depth_dir: str):
        depth_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            depth_files.extend(glob.glob(os.path.join(depth_dir, ext)))
        index = {}
        for p in sorted(depth_files):
            index[self._norm_key(p)] = p
            index[os.path.splitext(os.path.basename(p))[0].lower()] = p
        return index

    def _find_depth_for_rgb(self, rgb_path: str, depth_index: dict):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        candidates = [base.replace("_rgb", f"_{self.depth_folder_name}"), base.replace("rgb", self.depth_folder_name), base + f"_{self.depth_folder_name}", base, self._norm_key(rgb_path)]
        for cand in candidates:
            k = cand.lower()
            if k in depth_index:
                return depth_index[k]
        return None

    def _get_rotator(self, h: int, w: int):
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h = h
            self._rotator_w = w
        return self._rotator

    def _apply_full_pose3d(self, img_np, depth_np, valid_np):
        if not self.use_full_pose3d:
            return img_np, depth_np, valid_np
        yaw = random.uniform(0.0, self.pose_yaw_deg) if self.pose_yaw_deg > 0 else 0.0
        pitch = random.uniform(-self.pose_pitch_deg, self.pose_pitch_deg) if self.pose_pitch_deg > 0 else 0.0
        roll = random.uniform(-self.pose_roll_deg, self.pose_roll_deg) if self.pose_roll_deg > 0 else 0.0
        rgb_t = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
        depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
        valid_t = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
        rotator = self._get_rotator(img_np.shape[0], img_np.shape[1])
        rgb_rot = rotator.rotate_tensor(rgb_t, yaw=yaw, pitch=pitch, roll=roll, mode="bicubic")
        depth_rot = rotator.rotate_tensor(depth_t, yaw=yaw, pitch=pitch, roll=roll, mode="bilinear")
        valid_rot = rotator.rotate_tensor(valid_t, yaw=yaw, pitch=pitch, roll=roll, mode="nearest")
        img_np = (rgb_rot.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth_np = depth_rot.squeeze(0).cpu().numpy().astype(np.float32)
        valid_np = (valid_rot.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
        depth_np = np.clip(depth_np, 0.0, 1.0) * valid_np
        return img_np, depth_np, valid_np

    def __len__(self):
        return len(self.filenames)

    def _make_angles(self):
        angles = self.angle_centers.clone()
        if self.angle_jitter_deg > 0.0:
            jitter = torch.empty_like(angles).uniform_(-self.angle_jitter_deg, self.angle_jitter_deg)
            angles = angles + jitter
            lat_min = -90.0 + self.base_v_fov / 2.0
            lat_max = 90.0 - self.base_v_fov / 2.0
            angles[:, 1].clamp_(lat_min, lat_max)
        return angles.float()

    def _extract_patches_gpu(self, pano_tensor, angles_deg, h_fov_rad, v_fov_rad, mode="bicubic"):
        device = self.aug_device
        pano_tensor = pano_tensor.to(device, non_blocking=True)
        angles_t = angles_deg.to(device=device, dtype=torch.float32)
        N = angles_t.shape[0]
        theta = torch.deg2rad(angles_t[:, 0])
        phi = torch.deg2rad(angles_t[:, 1])
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        nc = torch.stack([cos_phi * cos_theta, -cos_phi * sin_theta, sin_phi], dim=1)
        xn = torch.stack([-sin_theta, -cos_theta, torch.zeros_like(theta)], dim=1)
        yn = torch.stack([-sin_phi * cos_theta, sin_phi * sin_theta, cos_phi], dim=1)
        fx = (self.patch_w / 2.0) / torch.tan(torch.tensor(h_fov_rad / 2.0, device=device))
        fy = (self.patch_h / 2.0) / torch.tan(torch.tensor(v_fov_rad / 2.0, device=device))
        uu = self._pixel_u.to(device)
        vv = self._pixel_v.to(device)
        pts = (uu.view(1, -1, 1) / fx) * xn.unsqueeze(1) + (vv.view(1, -1, 1) / fy) * yn.unsqueeze(1) + nc.unsqueeze(1)
        dirs = F.normalize(pts, dim=-1)
        px, py, pz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        theta_erp = torch.atan2(-py, px)
        phi_erp = torch.asin(torch.clamp(pz, -1.0, 1.0))
        grid_x = theta_erp / torch.pi
        grid_y = -phi_erp / (torch.pi / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, self.patch_h, self.patch_w, 2)
        pano_batch = pano_tensor.unsqueeze(0).expand(N, -1, -1, -1)
        return F.grid_sample(pano_batch, grid, mode=mode, padding_mode="border", align_corners=True)

    def __getitem__(self, idx):
        data = self.filenames[idx]
        try:
            img_pil = Image.open(data["rgb"]).convert("RGB")
            depth_pil = Image.open(data["depth"])
            if self.target_h is not None and self.target_w is not None and img_pil.size != (self.target_w, self.target_h):
                img_pil = img_pil.resize((self.target_w, self.target_h), Image.BICUBIC)
                depth_pil = depth_pil.resize((self.target_w, self.target_h), Image.NEAREST)
            if self.color_jitter is not None and random.random() < 0.8:
                img_pil = self.color_jitter(img_pil)
            if self.blur is not None and random.random() < 0.1:
                img_pil = self.blur(img_pil)
            img_np = np.array(img_pil)
            depth_raw = np.array(depth_pil).astype(np.float32)
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[:, :, 0]
            depth_meters = depth_raw / self.depth_scale
            raw_valid = depth_raw != float(self.invalid_depth_raw)
            valid_np = (np.isfinite(depth_meters) & raw_valid & (depth_meters >= self.depth_valid_min)).astype(np.float32)
            depth_np = np.zeros_like(depth_meters, dtype=np.float32)
            depth_np[valid_np > 0.5] = np.clip(depth_meters[valid_np > 0.5], 0.0, self.max_depth) / self.max_depth
            depth_np = depth_np.astype(np.float32)

            img_np, depth_np, valid_np = self._apply_full_pose3d(img_np, depth_np, valid_np)

            if self.use_horizontal_roll:
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1).copy()
                depth_np = np.roll(depth_np, roll_idx, axis=1).copy()
                valid_np = np.roll(valid_np, roll_idx, axis=1).copy()
            if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                img_np = np.flip(img_np, axis=1).copy()
                depth_np = np.flip(depth_np, axis=1).copy()
                valid_np = np.flip(valid_np, axis=1).copy()

            angles = self._make_angles()
            scale = random.uniform(0.8, 1.2) if self.multiscale_sampling else 1.0
            curr_h_fov_rad = np.radians(self.base_h_fov * scale)
            curr_v_fov_rad = np.radians(self.base_v_fov * scale)

            rgb_tensor = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
            depth_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
            valid_tensor = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
            rgb_patches = self._extract_patches_gpu(rgb_tensor, angles, curr_h_fov_rad, curr_v_fov_rad, mode="bicubic")
            depth_patches = self._extract_patches_gpu(depth_tensor, angles, curr_h_fov_rad, curr_v_fov_rad, mode="bilinear")
            valid_patches = self._extract_patches_gpu(valid_tensor, angles, curr_h_fov_rad, curr_v_fov_rad, mode="nearest")
            rgb_patches = rgb_patches.detach().cpu()
            valid_patches = (valid_patches > 0.5).float().detach().cpu()
            depth_patches = depth_patches.clamp(0.0, 1.0).detach().cpu() * valid_patches
            views = torch.stack([self.normalize(p) for p in rgb_patches], dim=0)
        except Exception as e:
            print(f"[Error] Loading {data.get('rgb', 'unknown')}: {e}")
            n = self.u_steps * self.v_steps
            return (
                torch.zeros(n, self.in_chans, self.patch_h, self.patch_w),
                self.angle_centers.clone(),
                torch.zeros(n, 1, self.patch_h, self.patch_w),
                torch.zeros(n, 1, self.patch_h, self.patch_w),
            )
        return views, angles, depth_patches, valid_patches


def build_depth_dataset(is_train, args):
    return Stanford2D3DDepthPBDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=getattr(args, "input_size", None),
        is_train=is_train,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
