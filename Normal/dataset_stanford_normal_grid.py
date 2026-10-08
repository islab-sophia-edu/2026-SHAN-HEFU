import os
import glob
import math
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
    """Full-ERP yaw/pitch/roll augmentation for RGB / scalar / vector targets."""

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

    @staticmethod
    def _rotation_matrix(yaw: float, pitch: float, roll: float, device):
        y_, p_, r_ = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor(
            [[np.cos(y_), -np.sin(y_), 0.0], [np.sin(y_), np.cos(y_), 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        Ry = torch.tensor(
            [[np.cos(p_), 0.0, np.sin(p_)], [0.0, 1.0, 0.0], [-np.sin(p_), 0.0, np.cos(p_)]],
            device=device,
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, np.cos(r_), -np.sin(r_)], [0.0, np.sin(r_), np.cos(r_)]],
            device=device,
            dtype=torch.float32,
        )
        return Rz @ Ry @ Rx

    def rotate_tensor(self, img_tensor: torch.Tensor, yaw=0.0, pitch=0.0, roll=0.0, mode="bicubic"):
        if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
            return img_tensor

        orig_device = img_tensor.device
        img_4d = img_tensor.to(self.device).unsqueeze(0)
        R = self._rotation_matrix(yaw, pitch, roll, self.device)

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

    def rotate_normal_tensor(self, normal_tensor: torch.Tensor, yaw=0.0, pitch=0.0, roll=0.0):
        """
        Spatially rotate an ERP normal map and rotate vector components into the
        augmented coordinate frame. The passive image transform samples input at
        d_in = R d_out, so row-vector normals transform as n_out = n_in @ R.
        """
        if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
            return F.normalize(normal_tensor, dim=0, p=2, eps=1e-6)
        orig_device = normal_tensor.device
        R = self._rotation_matrix(yaw, pitch, roll, self.device)
        sampled = self.rotate_tensor(normal_tensor, yaw=yaw, pitch=pitch, roll=roll, mode="bilinear").to(self.device)
        c, h, w = sampled.shape
        vec = sampled.view(3, -1).T
        vec_out = torch.matmul(vec, R)
        out = vec_out.T.view(3, h, w).to(orig_device)
        return F.normalize(out, dim=0, p=2, eps=1e-6)


class Stanford2D3DNormalGridDataset(Dataset):
    """
    Stanford 2D-3D-S normal dataset for grid-token GCTT ablation.

    This version deliberately removes ODI / tangent-plane tokenization.  RGB,
    optional rayXYZ, normal targets, and valid masks are produced by direct ERP
    grid patchification:

        [C,H,W] -> [grid_h * 2*grid_h, C, H/grid_h, W/(2*grid_h)]

    GCTT-compatible fields are still returned when --use_gctt is active, but the
    gauge angle is fixed to zero.  Random gauge jitter is not applied because no
    tangent patch content is rotated in this ablation.
    """

    def __init__(self, root_dir: str, grid_height: int = 16, img_size=None, is_train: bool = True,
                 debug_limit: Optional[int] = None, args=None):
        self.root_dir = root_dir
        self.is_train = bool(is_train)
        self.args = args
        self.v_steps = int(grid_height)
        self.u_steps = 2 * int(grid_height)
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps

        self.rgb_folder_name = getattr(args, "rgb_folder_name", "rgb") if args is not None else "rgb"
        self.normal_folder_name = getattr(args, "normal_folder_name", "normal") if args is not None else "normal"
        self.depth_folder_name = getattr(args, "depth_folder_name", "depth") if args is not None else "depth"
        self.normal_label_source = str(getattr(args, "normal_label_source", "depth") if args is not None else "depth").lower()
        if self.normal_label_source not in ("normal", "depth"):
            raise ValueError("--normal_label_source must be 'normal' or 'depth'.")
        self.depth_scale = float(getattr(args, "depth_scale", 512.0)) if args is not None else 512.0
        self.invalid_depth_raw = float(getattr(args, "invalid_depth_raw", 65535.0)) if args is not None else 65535.0
        self.depth_valid_min = float(getattr(args, "depth_valid_min", 0.05)) if args is not None else 0.05
        self.depth_valid_max = float(getattr(args, "depth_valid_max", 0.0)) if args is not None else 0.0
        self.depth_normal_smooth = float(getattr(args, "depth_normal_smooth", 0.0)) if args is not None else 0.0
        self.invalid_normal_value = int(getattr(args, "invalid_normal_value", 128)) if args is not None else 128
        self.invalid_normal_tolerance = float(getattr(args, "invalid_normal_tolerance", 1.0)) if args is not None else 1.0
        self.normal_axis_perm = self._parse_int_list(getattr(args, "normal_axis_perm", "0,1,2") if args is not None else "0,1,2", 3)
        self.normal_axis_sign = self._parse_float_list(getattr(args, "normal_axis_sign", "1,1,1") if args is not None else "1,1,1", 3)
        self.normal_local_y_sign = float(getattr(args, "normal_local_y_sign", 1.0)) if args is not None else 1.0
        self.normal_target_frame = str(getattr(args, "normal_target_frame", "global") if args is not None else "global").lower()
        if self.normal_target_frame not in ("global", "local"):
            raise ValueError("--normal_target_frame must be global or local")

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
            self.patch_h = self.target_h // self.v_steps
            self.patch_w = self.target_w // self.u_steps
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.patch_h, self.patch_w = int(img_size[0]), int(img_size[1])
        elif isinstance(img_size, int):
            self.patch_h = self.patch_w = int(img_size)
        else:
            self.patch_h = self.patch_w = 128

        self.in_chans = int(getattr(args, "in_chans", 3)) if args is not None else 3
        if self.in_chans not in (3, 6):
            raise ValueError("This grid normal dataset supports --in_chans 3 or 6. Use 6 for RGB + ERP-grid rayXYZ.")
        self.ray_coord = str(getattr(args, "ray_coord", "world") if args is not None else "world").lower()
        if self.ray_coord not in ("world", "local"):
            raise ValueError("--ray_coord must be 'world' or 'local'.")

        print(f"[{'Train' if self.is_train else 'Val'}] Loading Stanford2D3D Grid Normal")
        print(f"[{'Train' if self.is_train else 'Val'}] Root         : {self.root_dir}")
        print(f"[{'Train' if self.is_train else 'Val'}] Areas        : {self.areas}")
        print(f"[{'Train' if self.is_train else 'Val'}] Tokenization : ERP grid only; no ODI / no tangent-plane")
        print(f"[{'Train' if self.is_train else 'Val'}] Normal folder: {self.normal_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Depth folder : {self.depth_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Label source : {self.normal_label_source} ({'depth-derived ERP geometry' if self.normal_label_source == 'depth' else 'official Stanford normal PNG'})")
        print(f"[{'Train' if self.is_train else 'Val'}] Patch Size   : {self.patch_h}x{self.patch_w}")
        print(f"[{'Train' if self.is_train else 'Val'}] Axis perm/sign: {self.normal_axis_perm} / {self.normal_axis_sign}; local_y_sign={self.normal_local_y_sign}")
        print(f"[{'Train' if self.is_train else 'Val'}] Input chans : {self.in_chans} ({'RGB+ERP rayXYZ' if self.in_chans == 6 else 'RGB only'}), ray_coord={self.ray_coord}")
        print(f"[{'Train' if self.is_train else 'Val'}] Target frame: {self.normal_target_frame}")

        self.filenames = self._collect_pairs(debug_limit)
        print(f"[{'Train' if self.is_train else 'Val'}] Found {len(self.filenames)} RGB/normal pairs.")
        if len(self.filenames) == 0:
            raise ValueError("Stanford normal dataset is empty. Check --data_path/--val_data_path and pano/rgb + pano/normal/depth folders.")

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

        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) if bool(getattr(args, "use_color_jitter", True)) and self.is_train else None
        self.blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5)) if bool(getattr(args, "use_blur", True)) and self.is_train else None
        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", False)) and self.is_train
        self.pose_yaw_deg = float(getattr(args, "pose_yaw_deg", 360.0)) if args is not None else 360.0
        self.pose_pitch_deg = float(getattr(args, "pose_pitch_deg", 30.0)) if args is not None else 30.0
        self.pose_roll_deg = float(getattr(args, "pose_roll_deg", 30.0)) if args is not None else 30.0
        self.use_horizontal_roll = bool(getattr(args, "use_horizontal_roll", True)) and self.is_train
        self.hflip_prob = float(getattr(args, "hflip_prob", 0.0)) if self.is_train and args is not None else 0.0
        if self.hflip_prob > 0.0:
            print("Warning: hflip for normal estimation changes handedness. It is supported, but disabled by default.")

        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._erp_xyz_cache = None
        self._erp_xyz_cache_size = None

    @staticmethod
    def _parse_int_list(s, n):
        vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
        if len(vals) != n or sorted(vals) != list(range(n)):
            raise ValueError(f"Expected a permutation of 0..{n-1}, got {s}")
        return vals

    @staticmethod
    def _parse_float_list(s, n):
        vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
        if len(vals) != n:
            raise ValueError(f"Expected {n} comma-separated signs/scales, got {s}")
        return vals

    def _collect_pairs(self, debug_limit: Optional[int]):
        pairs = []
        for area in self.areas:
            rgb_dir = os.path.join(self.root_dir, area, "pano", self.rgb_folder_name)
            normal_dir = os.path.join(self.root_dir, area, "pano", self.normal_folder_name)
            depth_dir = os.path.join(self.root_dir, area, "pano", self.depth_folder_name)
            if not os.path.isdir(rgb_dir):
                print(f"[Skip] {area}: missing {rgb_dir}")
                continue
            if self.normal_label_source == "normal" and not os.path.isdir(normal_dir):
                print(f"[Skip] {area}: missing {normal_dir}")
                continue
            if self.normal_label_source == "depth" and not os.path.isdir(depth_dir):
                print(f"[Skip] {area}: missing {depth_dir}")
                continue
            rgb_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
            normal_index = self._build_normal_index(normal_dir) if os.path.isdir(normal_dir) else {}
            depth_index = self._build_depth_index(depth_dir) if os.path.isdir(depth_dir) else {}
            area_pairs = 0
            for rgb_path in sorted(rgb_files):
                normal_path = self._find_normal_for_rgb(rgb_path, normal_index) if normal_index else None
                depth_path = self._find_depth_for_rgb(rgb_path, depth_index) if depth_index else None
                ok = (normal_path is not None) if self.normal_label_source == "normal" else (depth_path is not None)
                if ok:
                    pairs.append({"rgb": rgb_path, "normal": normal_path, "depth": depth_path, "area": area})
                    area_pairs += 1
            print(f"[Area] {area}: RGB={len(rgb_files)}, paired={area_pairs}")
        if debug_limit:
            pairs = pairs[:debug_limit]
        return pairs

    @staticmethod
    def _norm_key(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        drop = {"rgb", "rgba", "color", "colour", "pano", "panorama", "normal", "normals", "norm", "depth", "depths"}
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t and t not in drop]
        return "_".join(tokens)

    def _build_normal_index(self, normal_dir: str):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(normal_dir, ext)))
        index = {}
        for p in sorted(files):
            index[self._norm_key(p)] = p
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            index[stem] = p
        return index

    def _find_normal_for_rgb(self, rgb_path: str, normal_index: dict):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        candidates = [
            base.replace("_rgb", "_normals"),
            base.replace("_rgb", "_normal"),
            base.replace("rgb", "normals"),
            base.replace("rgb", "normal"),
            base + "_normals",
            base + "_normal",
            base,
            self._norm_key(rgb_path),
        ]
        for cand in candidates:
            k = cand.lower()
            if k in normal_index:
                return normal_index[k]
        return None

    def _build_depth_index(self, depth_dir: str):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(depth_dir, ext)))
        index = {}
        for p in sorted(files):
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            index[self._norm_key(p)] = p
            index[stem] = p
        return index

    def _find_depth_for_rgb(self, rgb_path: str, depth_index: dict):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        candidates = [
            base.replace("_rgb", "_depth"),
            base.replace("rgb", "depth"),
            base + "_depth",
            base,
            self._norm_key(rgb_path),
        ]
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

    def _decode_normal(self, normal_pil: Image.Image):
        raw = np.array(normal_pil.convert("RGB")).astype(np.float32)
        invalid = np.all(np.abs(raw - float(self.invalid_normal_value)) <= self.invalid_normal_tolerance, axis=2)
        valid = (~invalid).astype(np.float32)

        n = (raw - 127.5) / 127.5
        n = n[..., self.normal_axis_perm]
        signs = np.array(self.normal_axis_sign, dtype=np.float32).reshape(1, 1, 3)
        n = n * signs
        norm = np.linalg.norm(n, axis=2, keepdims=True)
        valid = valid * np.isfinite(norm[..., 0]).astype(np.float32) * (norm[..., 0] > 1e-4).astype(np.float32)
        n = n / (norm + 1e-8)
        n[valid <= 0.5] = 0.0
        return n.astype(np.float32), valid.astype(np.float32)

    def _decode_depth(self, depth_pil: Image.Image):
        raw = np.array(depth_pil).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        depth_m = raw / max(self.depth_scale, 1e-8)
        valid = np.isfinite(depth_m) & (np.abs(raw - self.invalid_depth_raw) > 0.5) & (depth_m >= self.depth_valid_min)
        if self.depth_valid_max and self.depth_valid_max > 0:
            valid = valid & (depth_m <= self.depth_valid_max)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[~valid] = 0.0
        return depth_m.astype(np.float32), valid.astype(np.float32)

    def _apply_full_pose3d_normal(self, img_np, normal_np, valid_np):
        if not self.use_full_pose3d:
            return img_np, normal_np, valid_np
        yaw = random.uniform(0.0, self.pose_yaw_deg) if self.pose_yaw_deg > 0 else 0.0
        pitch = random.uniform(-self.pose_pitch_deg, self.pose_pitch_deg) if self.pose_pitch_deg > 0 else 0.0
        roll = random.uniform(-self.pose_roll_deg, self.pose_roll_deg) if self.pose_roll_deg > 0 else 0.0

        rgb_t = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
        normal_t = torch.from_numpy(normal_np).float().permute(2, 0, 1)
        valid_t = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
        rotator = self._get_rotator(img_np.shape[0], img_np.shape[1])
        rgb_rot = rotator.rotate_tensor(rgb_t, yaw=yaw, pitch=pitch, roll=roll, mode="bicubic")
        normal_rot = rotator.rotate_normal_tensor(normal_t, yaw=yaw, pitch=pitch, roll=roll)
        valid_rot = rotator.rotate_tensor(valid_t, yaw=yaw, pitch=pitch, roll=roll, mode="nearest")
        img_np = (rgb_rot.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        normal_np = normal_rot.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        valid_np = (valid_rot.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
        normal_np[valid_np <= 0.5] = 0.0
        return img_np, normal_np, valid_np

    def _apply_full_pose3d_depth(self, img_np, depth_m_np, valid_np):
        if not self.use_full_pose3d:
            return img_np, depth_m_np, valid_np
        yaw = random.uniform(0.0, self.pose_yaw_deg) if self.pose_yaw_deg > 0 else 0.0
        pitch = random.uniform(-self.pose_pitch_deg, self.pose_pitch_deg) if self.pose_pitch_deg > 0 else 0.0
        roll = random.uniform(-self.pose_roll_deg, self.pose_roll_deg) if self.pose_roll_deg > 0 else 0.0
        rgb_t = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
        depth_t = torch.from_numpy(depth_m_np.astype(np.float32)).unsqueeze(0)
        valid_t = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
        rotator = self._get_rotator(img_np.shape[0], img_np.shape[1])
        rgb_rot = rotator.rotate_tensor(rgb_t, yaw=yaw, pitch=pitch, roll=roll, mode="bicubic")
        depth_rot = rotator.rotate_tensor(depth_t, yaw=yaw, pitch=pitch, roll=roll, mode="bilinear")
        valid_rot = rotator.rotate_tensor(valid_t, yaw=yaw, pitch=pitch, roll=roll, mode="nearest")
        img_np = (rgb_rot.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        depth_m_np = depth_rot.squeeze(0).cpu().numpy().astype(np.float32)
        valid_np = (valid_rot.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
        depth_m_np[valid_np <= 0.5] = 0.0
        return img_np, depth_m_np, valid_np

    def __len__(self):
        return len(self.filenames)

    def _make_angles_and_gauges(self):
        angles = self.angle_centers.clone()
        gauge_angles = torch.zeros(angles.shape[0], 1, dtype=torch.float32)
        return angles.float(), gauge_angles

    def _patchify_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [C,H,W]
        c, h, w = tensor.shape
        expected_h = self.v_steps * self.patch_h
        expected_w = self.u_steps * self.patch_w
        if h != expected_h or w != expected_w:
            raise ValueError(f"Unexpected tensor size {h}x{w}; expected {expected_h}x{expected_w}.")
        patches = tensor.view(c, self.v_steps, self.patch_h, self.u_steps, self.patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()
        return patches.view(self.v_steps * self.u_steps, c, self.patch_h, self.patch_w)

    def _erp_xyz(self, h: int, w: int) -> torch.Tensor:
        cache_key = (int(h), int(w))
        if self._erp_xyz_cache is not None and self._erp_xyz_cache_size == cache_key:
            return self._erp_xyz_cache
        v, u = torch.meshgrid(
            torch.linspace(1, -1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        theta = u * math.pi
        phi = v * math.pi / 2.0
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

    def _compute_global_normals_from_depth_erp(self, depth_m: torch.Tensor, valid: torch.Tensor):
        """
        Derive global/PanoMAE-frame normals from an ERP metric depth map.
        depth_m: [H,W], radial distance in meters. valid: [H,W].
        """
        device = depth_m.device
        h, w = depth_m.shape
        rays = self._erp_xyz(h, w).to(device=device, dtype=torch.float32)
        points = rays * depth_m.float().unsqueeze(0)
        valid_b = valid.float().unsqueeze(0)
        if self.depth_normal_smooth > 0:
            k = int(max(3, round(self.depth_normal_smooth)))
            if k % 2 == 0:
                k += 1
            points = F.avg_pool2d(points.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
        # horizontal derivative uses circular wrap; vertical derivative does not.
        dx = torch.roll(points, shifts=-1, dims=2) - torch.roll(points, shifts=1, dims=2)
        dy = points[:, 2:, :] - points[:, :-2, :]
        dy = F.pad(dy, (0, 0, 1, 1), mode="replicate")
        # dy points image-down because row index increases downward.  -cross(dx,dy)
        # gives camera-facing normals for front-facing visible surfaces.
        n = -torch.cross(dx.permute(1, 2, 0), dy.permute(1, 2, 0), dim=-1).permute(2, 0, 1)
        n = F.normalize(n, dim=0, p=2, eps=1e-6)
        dot = (n * rays).sum(dim=0, keepdim=True)
        n = torch.where(dot > 0, -n, n)

        valid_x = torch.roll(valid_b, shifts=-1, dims=2) * torch.roll(valid_b, shifts=1, dims=2)
        valid_y = torch.zeros_like(valid_b)
        valid_y[:, 1:-1, :] = valid_b[:, :-2, :] * valid_b[:, 2:, :]
        normal_valid = (valid_b > 0.5) & (valid_x > 0.5) & (valid_y > 0.5) & torch.isfinite(n).all(dim=0, keepdim=True)
        n = n * normal_valid.float()
        return n.float(), normal_valid.float()

    @staticmethod
    def _frames_from_angles_cpu(angles_deg: torch.Tensor, gauge_angles_deg: Optional[torch.Tensor] = None):
        angles_t = angles_deg.float()
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
            gauge = torch.zeros(angles_t.shape[0], dtype=torch.float32)
        else:
            gauge = gauge_angles_deg.float().view(-1)
        psi = torch.deg2rad(gauge)
        c = torch.cos(psi).unsqueeze(1)
        s = torch.sin(psi).unsqueeze(1)
        xg = F.normalize(c * xn + s * yn, dim=1)
        yg = F.normalize(-s * xn + c * yn, dim=1)
        nc = F.normalize(nc, dim=1)
        return nc, xg, yg

    def _world_normal_patches_to_local(self, normal_patches, angles, gauge_angles):
        nc, xg, yg = self._frames_from_angles_cpu(angles, gauge_angles)
        y_basis = self.normal_local_y_sign * yg
        basis = torch.stack([xg, y_basis, nc], dim=1)  # [N, local_axis, global_dim]
        local = torch.einsum("nchw,nkc->nkhw", normal_patches, basis)
        return F.normalize(local, dim=1, p=2, eps=1e-6)

    def _make_ray_patches(self, h: int, w: int, angles: torch.Tensor, gauge_angles: torch.Tensor):
        rays_global = self._patchify_grid(self._erp_xyz(h, w))
        if self.ray_coord == "world":
            return rays_global.float()
        # Convert per-pixel global rays to each token's local center frame.
        return self._world_normal_patches_to_local(rays_global, angles, gauge_angles).float()

    def __getitem__(self, idx: int):
        data = self.filenames[idx]
        try:
            img_pil = Image.open(data["rgb"]).convert("RGB")
            normal_pil = Image.open(data["normal"]).convert("RGB") if data.get("normal") is not None else None
            depth_pil = Image.open(data["depth"]) if data.get("depth") is not None else None
            if self.target_h is not None and self.target_w is not None:
                if img_pil.size != (self.target_w, self.target_h):
                    img_pil = img_pil.resize((self.target_w, self.target_h), Image.BICUBIC)
                    if normal_pil is not None:
                        normal_pil = normal_pil.resize((self.target_w, self.target_h), Image.NEAREST)
                    if depth_pil is not None:
                        depth_pil = depth_pil.resize((self.target_w, self.target_h), Image.NEAREST)

            if self.color_jitter is not None and random.random() < 0.8:
                img_pil = self.color_jitter(img_pil)
            if self.blur is not None and random.random() < 0.1:
                img_pil = self.blur(img_pil)

            img_np = np.array(img_pil)

            if self.normal_label_source == "normal":
                normal_np, valid_np = self._decode_normal(normal_pil)
                img_np, normal_np, valid_np = self._apply_full_pose3d_normal(img_np, normal_np, valid_np)
                if self.use_horizontal_roll:
                    roll_idx = random.randint(0, img_np.shape[1] - 1)
                    img_np = np.roll(img_np, roll_idx, axis=1).copy()
                    normal_np = np.roll(normal_np, roll_idx, axis=1).copy()
                    valid_np = np.roll(valid_np, roll_idx, axis=1).copy()
                if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                    img_np = np.flip(img_np, axis=1).copy()
                    normal_np = np.flip(normal_np, axis=1).copy()
                    valid_np = np.flip(valid_np, axis=1).copy()
                    normal_np[..., 0] *= -1.0
                normal_tensor = torch.from_numpy(normal_np.astype(np.float32)).permute(2, 0, 1)
                valid_tensor = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
            else:
                if depth_pil is None:
                    raise FileNotFoundError(f"Depth target missing for {data['rgb']}")
                depth_m_np, depth_valid_np = self._decode_depth(depth_pil)
                img_np, depth_m_np, depth_valid_np = self._apply_full_pose3d_depth(img_np, depth_m_np, depth_valid_np)
                if self.use_horizontal_roll:
                    roll_idx = random.randint(0, img_np.shape[1] - 1)
                    img_np = np.roll(img_np, roll_idx, axis=1).copy()
                    depth_m_np = np.roll(depth_m_np, roll_idx, axis=1).copy()
                    depth_valid_np = np.roll(depth_valid_np, roll_idx, axis=1).copy()
                if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                    img_np = np.flip(img_np, axis=1).copy()
                    depth_m_np = np.flip(depth_m_np, axis=1).copy()
                    depth_valid_np = np.flip(depth_valid_np, axis=1).copy()
                depth_tensor = torch.from_numpy(depth_m_np.astype(np.float32))
                valid_depth_tensor = torch.from_numpy(depth_valid_np.astype(np.float32))
                normal_tensor, valid_tensor = self._compute_global_normals_from_depth_erp(depth_tensor, valid_depth_tensor)

            angles, gauge_angles = self._make_angles_and_gauges()

            rgb_tensor = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
            rgb_patches = self._patchify_grid(rgb_tensor)
            rgb_views = torch.stack([self.normalize(p) for p in rgb_patches], dim=0)

            normal_patches = self._patchify_grid(normal_tensor.float())
            valid_patches = (self._patchify_grid(valid_tensor.float()) > 0.5).float()
            normal_patches = F.normalize(normal_patches, dim=1, p=2, eps=1e-6) * valid_patches

            if self.normal_target_frame == "local":
                normal_patches = self._world_normal_patches_to_local(normal_patches, angles, gauge_angles) * valid_patches

            if self.in_chans == 6:
                ray_patches = self._make_ray_patches(rgb_tensor.shape[1], rgb_tensor.shape[2], angles, gauge_angles)
                views = torch.cat([rgb_views, ray_patches], dim=1)
            else:
                views = rgb_views

        except Exception as e:
            print(f"[Error] Loading {data.get('rgb', 'unknown')}: {e}")
            n = self.u_steps * self.v_steps
            views = torch.zeros(n, self.in_chans, self.patch_h, self.patch_w)
            angles = self.angle_centers.clone()
            gauge_angles = torch.zeros(n, 1)
            normal_patches = torch.zeros(n, 3, self.patch_h, self.patch_w)
            valid_patches = torch.zeros(n, 1, self.patch_h, self.patch_w)

        if self.use_gctt:
            return views, angles, gauge_angles, normal_patches, valid_patches
        return views, angles, normal_patches, valid_patches


def build_normal_dataset(is_train, args):
    return Stanford2D3DNormalGridDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=getattr(args, "input_size", None),
        is_train=is_train,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
