"""GCTT normal ablation: official Stanford pano/normal GT only (train/val)."""
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
from normal_pose_frame import pose_path_from_normal, world_normals_to_erp
from PIL import Image
from torchvision import transforms


def _sample_periodic_erp(pano_batch, theta, phi, mode="bilinear"):
    """ERP center-aligned sampling with periodic longitude and clamped latitude."""
    batch, channels, height, width = pano_batch.shape
    source = pano_batch[:1]
    padded = torch.cat((source[..., -2:], source, source[..., :2]), dim=-1)
    padded = padded.expand(batch, -1, -1, -1)
    px = ((theta / (2.0 * math.pi) + 0.5) * width - 0.5).remainder(width)
    gx = 2.0 * (px + 2.5) / (width + 4.0) - 1.0
    gy = -2.0 * phi / math.pi
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(padded, grid, mode=mode, padding_mode="border", align_corners=False)


class ERPRotator:
    """Full-ERP yaw/pitch/roll augmentation for RGB / vector targets."""

    def __init__(self, h: int, w: int, device: str = "cpu"):
        self.h = int(h)
        self.w = int(w)
        self.device = torch.device(device)

        row, col = torch.meshgrid(
            torch.arange(self.h, device=self.device, dtype=torch.float32),
            torch.arange(self.w, device=self.device, dtype=torch.float32), indexing="ij",
        )
        theta = (2.0 * (col + 0.5) / self.w - 1.0) * math.pi
        phi = (0.5 - (row + 0.5) / self.h) * math.pi
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
        theta = torch.atan2(-xyz_rot[:, 1], xyz_rot[:, 0]).view(1, self.h, self.w)
        phi = torch.asin(xyz_rot[:, 2].clamp(-1.0, 1.0)).view(1, self.h, self.w)
        rotated = _sample_periodic_erp(img_4d, theta, phi, mode=mode)
        return rotated.squeeze(0).to(orig_device)

    def rotate_normal_tensor(self, normal_tensor: torch.Tensor, yaw=0.0, pitch=0.0, roll=0.0):
        """
        Spatially rotate an ERP normal map and rotate vector components into the
        augmented coordinate frame.  The passive image transform samples input at
        d_in = R d_out, so vectors are transformed by R^T.
        """
        if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
            return F.normalize(normal_tensor, dim=0, p=2, eps=1e-6)
        orig_device = normal_tensor.device
        R = self._rotation_matrix(yaw, pitch, roll, self.device)
        sampled = self.rotate_tensor(normal_tensor, yaw=yaw, pitch=pitch, roll=roll, mode="bilinear").to(self.device)
        c, h, w = sampled.shape
        vec = sampled.view(3, -1).T
        vec_out = torch.matmul(vec, R)  # row-vector n_out = n_in @ R == R^T n_in in column form
        out = vec_out.T.view(3, h, w).to(orig_device)
        return F.normalize(out, dim=0, p=2, eps=1e-6)


class Stanford2D3DNormalGCTTDataset(Dataset):
    """
    Stanford 2D-3D-S normal dataset for the GCTT normal engine.

    Expected directory layout:
        root_dir/area_x/pano/rgb/*.png
        root_dir/area_x/pano/normal/*.png

    Returns, before DataLoader batching:
        views        : [N, 3, patch_h, patch_w], ImageNet-normalized RGB
        angles       : [N, 2], degrees, [lon/theta, lat/phi]
        gauge_angles : [N, 1], degrees, tangent-frame gauge psi_i
        normals      : [N, 3, patch_h, patch_w], unit normals in Pose-aligned ERP/ODI frame
        valid_mask   : [N, 1, patch_h, patch_w], 1 for valid normal pixels

    Normal convention:
        Stanford normal PNGs are 127.5-centered RGB normal images; #808080 is missing mesh.
        This dataset first converts RGB -> normal vector, applies a configurable
        axis permutation/sign to Stanford World XYZ, then transforms each normal
        using the corresponding official panorama Pose into PanoMAE ERP coordinates.
        Legacy engine name 'global' refers to its fixed ERP/ODI 3D axes.
    """

    def __init__(self, root_dir: str, grid_height: int = 16, img_size=None, is_train: bool = True,
                 debug_limit: Optional[int] = None, args=None):
        self.grid_tokenization = getattr(args, "variant", "gctt") in {"grid", "mae"}
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
        self.normal_label_source = str(getattr(args, "normal_label_source", "normal") if args is not None else "normal").lower()
        if self.normal_label_source != "normal":
            raise ValueError("Normal benchmark uses official pano/normal GT for BOTH train and val; set --normal_label_source normal")
        self.depth_scale = float(getattr(args, "depth_scale", 512.0)) if args is not None else 512.0
        self.invalid_depth_raw = float(getattr(args, "invalid_depth_raw", 65535.0)) if args is not None else 65535.0
        self.depth_valid_min = float(getattr(args, "depth_valid_min", 0.05)) if args is not None else 0.05
        self.depth_valid_max = float(getattr(args, "depth_valid_max", 0.0)) if args is not None else 0.0
        self.depth_normal_smooth = float(getattr(args, "depth_normal_smooth", 0.0)) if args is not None else 0.0
        self.invalid_normal_value = int(getattr(args, "invalid_normal_value", 128)) if args is not None else 128
        self.invalid_normal_tolerance = float(getattr(args, "invalid_normal_tolerance", 1.0)) if args is not None else 1.0
        self.normal_axis_perm = self._parse_int_list(getattr(args, "normal_axis_perm", "0,2,1") if args is not None else "0,2,1", 3)
        self.normal_axis_sign = self._parse_float_list(getattr(args, "normal_axis_sign", "-1,-1,1") if args is not None else "-1,-1,1", 3)
        if any(abs(v) != 1.0 for v in self.normal_axis_sign):
            raise ValueError("Official Normal axis signs must be +/-1 to preserve angular geometry")
        self.normal_local_y_sign = float(getattr(args, "normal_local_y_sign", 1.0)) if args is not None else 1.0
        self.normal_target_frame = str(getattr(args, "normal_target_frame", "global") if args is not None else "global").lower()
        self.official_normal_coordinate = str(getattr(args, "official_normal_coordinate", "erp")).lower()
        if self.official_normal_coordinate != "erp":
            raise ValueError("Only Pose-aligned ERP normal supervision is supported")
        if self.normal_target_frame != "global":
            raise ValueError("Official normal GT benchmark requires --normal_target_frame global")

        if self.is_train:
            areas_str = getattr(args, "train_areas", "area_1,area_2,area_3,area_4,area_6") if args else "area_1,area_2,area_3,area_4,area_6"
        else:
            areas_str = getattr(args, "val_areas", "area_5a,area_5b") if args else "area_5a,area_5b"
        self.areas = [a.strip() for a in areas_str.split(",") if a.strip()]
        if not self.areas:
            raise ValueError("At least one Stanford area is required")

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
        if self.in_chans != 3:
            raise ValueError("This clean Normal benchmark requires RGB-only --in_chans 3")
        self.ray_coord = str(getattr(args, "ray_coord", "world") if args is not None else "world").lower()
        if self.ray_coord not in ("world", "local"):
            raise ValueError("--ray_coord must be 'world' or 'local'.")

        print(f"[{'Train' if self.is_train else 'Val'}] Stanford2D3D Normal variant={getattr(args, 'variant', 'gctt')}")
        print(f"[{'Train' if self.is_train else 'Val'}] Root         : {self.root_dir}")
        print(f"[{'Train' if self.is_train else 'Val'}] Areas        : {self.areas}")
        print(f"[{'Train' if self.is_train else 'Val'}] Normal folder: {self.normal_folder_name}")
        print(f"[{'Train' if self.is_train else 'Val'}] Label source : official Stanford pano/normal PNG")
        print(f"[{'Train' if self.is_train else 'Val'}] Patch Size   : {self.patch_h}x{self.patch_w}")
        print(f"[{'Train' if self.is_train else 'Val'}] Axis perm/sign: {self.normal_axis_perm} / {self.normal_axis_sign}; local_y_sign={self.normal_local_y_sign} (default +1 => GCTT frame [x_g,y_g,n_c])")
        print(f"[{'Train' if self.is_train else 'Val'}] Input chans : RGB only")
        print(f"[{'Train' if self.is_train else 'Val'}] Target frame: ERP (Pose-aligned official world normals; engine legacy name={self.normal_target_frame})")

        self.filenames = self._collect_pairs(debug_limit)
        print(f"[{'Train' if self.is_train else 'Val'}] Found {len(self.filenames)} RGB/normal pairs.")
        if len(self.filenames) == 0:
            raise ValueError("Stanford normal dataset is empty. Check --data_path/--val_data_path and pano/rgb + pano/normal folders.")

        v_centers = torch.linspace(90.0 - self.v_fov / 2.0, -90.0 + self.v_fov / 2.0, self.v_steps)
        u_centers = torch.linspace(-180.0, 180.0, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing="ij")
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).float()

        self.use_gctt = getattr(args, "variant", "gctt") in {"gctt", "noadaptive", "grid"}
        self.gctt_gauge_jitter_deg = float(getattr(args, "gctt_gauge_jitter_deg", 30.0)) if self.is_train else 0.0
        self.gctt_local_gauge_jitter_deg = float(getattr(args, "gctt_local_gauge_jitter_deg", 0.0)) if self.is_train else 0.0
        self.angle_jitter_deg = float(getattr(args, "angle_jitter_deg", 0.0)) if self.is_train else 0.0

        aug_device = getattr(args, "aug_device", "cuda") if args is not None else "cuda"
        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else "cpu")

        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) if bool(getattr(args, "use_color_jitter", True)) and self.is_train else None
        self.blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5)) if bool(getattr(args, "use_blur", True)) and self.is_train else None
        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", True)) and self.is_train
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

        u_lin = torch.linspace(-(self.patch_w - 1) / 2.0, (self.patch_w - 1) / 2.0, self.patch_w)
        v_lin = torch.linspace((self.patch_h - 1) / 2.0, -(self.patch_h - 1) / 2.0, self.patch_h)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing="xy")
        self._pixel_u = uu.reshape(-1)
        self._pixel_v = vv.reshape(-1)

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
            if not os.path.isdir(rgb_dir):
                raise FileNotFoundError(f"Required RGB area is missing: {rgb_dir}")
            if not os.path.isdir(normal_dir):
                raise FileNotFoundError(f"Required official normal GT area is missing: {normal_dir}")
            rgb_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
            normal_index = self._build_normal_index(normal_dir) if os.path.isdir(normal_dir) else {}
            area_pairs = 0
            for rgb_path in sorted(rgb_files):
                normal_path = self._find_normal_for_rgb(rgb_path, normal_index) if normal_index else None
                ok = normal_path is not None
                if ok:
                    pose_path = pose_path_from_normal(normal_path)
                    pairs.append({"rgb": rgb_path, "normal": normal_path, "pose": str(pose_path), "area": area})
                    area_pairs += 1
            print(f"[Area] {area}: RGB={len(rgb_files)}, paired={area_pairs}")
        if debug_limit:
            pairs = pairs[:debug_limit]
        return pairs

    @staticmethod
    def _norm_key(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        drop = {"rgb", "rgba", "color", "colour", "pano", "panorama", "normal", "normals", "norm"}
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

    def _apply_full_pose3d(self, img_np, normal_np, valid_np):
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
            local = torch.empty(angles.shape[0], 1).uniform_(-self.gctt_local_gauge_jitter_deg, self.gctt_local_gauge_jitter_deg)
        else:
            local = torch.zeros(angles.shape[0], 1)
        gauge_angles = torch.full((angles.shape[0], 1), float(global_gauge)) + local
        if not self.use_gctt:
            gauge_angles.zero_()
        return angles.float(), gauge_angles.float()

    def _frames_from_angles(self, angles_deg: torch.Tensor, gauge_angles_deg: Optional[torch.Tensor] = None):
        device = self.aug_device
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
        xg = F.normalize(c * xn + s * yn, dim=1)
        yg = F.normalize(-s * xn + c * yn, dim=1)
        nc = F.normalize(nc, dim=1)
        return nc, xg, yg

    def _extract_patches_gpu(self, pano_tensor, angles_deg, h_fov_rad, v_fov_rad, gauge_angles_deg=None, mode="bicubic"):
        device = self.aug_device
        pano_tensor = pano_tensor.to(device, non_blocking=True)
        angles_t = angles_deg.to(device=device, dtype=torch.float32)
        nc, xg, yg = self._frames_from_angles(angles_t, gauge_angles_deg)
        N = angles_t.shape[0]
        fx = (self.patch_w / 2.0) / torch.tan(torch.tensor(h_fov_rad / 2.0, device=device))
        fy = (self.patch_h / 2.0) / torch.tan(torch.tensor(v_fov_rad / 2.0, device=device))
        uu = self._pixel_u.to(device)
        vv = self._pixel_v.to(device)
        pts = (uu.view(1, -1, 1) / fx) * xg.unsqueeze(1) + (vv.view(1, -1, 1) / fy) * yg.unsqueeze(1) + nc.unsqueeze(1)
        dirs = F.normalize(pts, dim=-1)
        px, py, pz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        theta_erp = torch.atan2(-py, px)
        phi_erp = torch.asin(torch.clamp(pz, -1.0, 1.0))
        theta_erp = theta_erp.view(N, self.patch_h, self.patch_w)
        phi_erp = phi_erp.view(N, self.patch_h, self.patch_w)
        pano_batch = pano_tensor.unsqueeze(0).expand(N, -1, -1, -1)
        patches = _sample_periodic_erp(pano_batch, theta_erp, phi_erp, mode=mode)
        return patches

    def _patchify_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        channels, height, width = tensor.shape
        if (height, width) != (self.target_h, self.target_w):
            raise ValueError("Unexpected ERP grid resolution")
        if height % self.v_steps or width % self.u_steps:
            raise ValueError("ERP resolution is not divisible by grid")
        return (tensor.contiguous()
                .view(channels, self.v_steps, self.patch_h, self.u_steps, self.patch_w)
                .permute(1, 3, 0, 2, 4).contiguous()
                .view(self.v_steps * self.u_steps, channels, self.patch_h, self.patch_w))

    def __getitem__(self, idx: int):
        item = self.filenames[idx]
        try:
            img_pil = Image.open(item["rgb"]).convert("RGB")
            normal_pil = Image.open(item["normal"]).convert("RGB")
            if self.target_h is not None and self.target_w is not None:
                wanted = (self.target_w, self.target_h)
                if img_pil.size != wanted:
                    img_pil = img_pil.resize(wanted, Image.Resampling.BICUBIC)
                if normal_pil.size != wanted:
                    normal_pil = normal_pil.resize(wanted, Image.Resampling.NEAREST)
            if self.color_jitter is not None and random.random() < 0.8:
                img_pil = self.color_jitter(img_pil)
            if self.blur is not None and random.random() < 0.1:
                img_pil = self.blur(img_pil)
            rgb = np.asarray(img_pil)
            world_normal, valid = self._decode_normal(normal_pil)
            erp_normal = world_normals_to_erp(world_normal, item["pose"])
            erp_normal *= valid[..., None]
            rgb, world_normal, valid = self._apply_full_pose3d(rgb, erp_normal, valid)
            angles, gauge_angles = self._make_angles_and_gauges()
            if self.use_horizontal_roll:
                if self.grid_tokenization:
                    # A grid experiment keeps token centers fixed: rotate the ERP
                    # image AND world-space normal vectors coherently instead.
                    yaw = random.uniform(-180.0, 180.0)
                    old = self.use_full_pose3d
                    self.use_full_pose3d = True
                    try:
                        # Explicit yaw rotation with consistent vector rotation.
                        rotator = self._get_rotator(rgb.shape[0], rgb.shape[1])
                        rt = torch.from_numpy(rgb.copy()).float().div(255.0).permute(2, 0, 1)
                        nt = torch.from_numpy(world_normal.copy()).float().permute(2, 0, 1)
                        vt = torch.from_numpy(valid.copy()).float().unsqueeze(0)
                        rgb_t = rotator.rotate_tensor(rt, yaw=yaw, mode="bicubic").clamp(0, 1)
                        n_t = rotator.rotate_normal_tensor(nt, yaw=yaw)
                        v_t = (rotator.rotate_tensor(vt, yaw=yaw, mode="nearest") > 0.5).float()
                        rgb = (rgb_t.permute(1, 2, 0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
                        world_normal = (n_t * v_t).permute(1,2,0).cpu().numpy().astype(np.float32)
                        valid = v_t[0].cpu().numpy()
                    finally:
                        self.use_full_pose3d = old
                else:
                    yaw = random.uniform(-180.0, 180.0)
                    angles[:, 0] = ((angles[:, 0] + yaw + 180) % 360) - 180
            if self.hflip_prob > 0 and random.random() < self.hflip_prob:
                rgb = np.flip(rgb, axis=1).copy()
                world_normal = np.flip(world_normal, axis=1).copy()
                valid = np.flip(valid, axis=1).copy()
                world_normal[..., 1] *= -1.0  # ERP longitude reflection flips world Y.
            rgb_tensor = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1) / 255.0
            normal_tensor = torch.from_numpy(world_normal.copy()).float().permute(2, 0, 1)
            valid_tensor = torch.from_numpy(valid.copy()).float().unsqueeze(0)
            if self.grid_tokenization:
                rgb_patches = self._patchify_grid(rgb_tensor)
                normal_patches = self._patchify_grid(normal_tensor * valid_tensor)
                valid_patches = (self._patchify_grid(valid_tensor) > 0.5).float()
                normals = F.normalize(normal_patches.float(), dim=1, eps=1e-6) * valid_patches
            else:
                gauge = gauge_angles if self.use_gctt else None
                vfov, hfov = np.radians(self.v_fov), np.radians(self.h_fov)
                rgb_patches = self._extract_patches_gpu(rgb_tensor, angles, hfov, vfov,
                                                        gauge_angles_deg=gauge, mode="bicubic").clamp(0, 1)
                n_num = self._extract_patches_gpu(normal_tensor * valid_tensor, angles, hfov, vfov,
                                                   gauge_angles_deg=gauge, mode="bilinear")
                n_den = self._extract_patches_gpu(valid_tensor, angles, hfov, vfov,
                                                   gauge_angles_deg=gauge, mode="bilinear")
                valid_patches = self._extract_patches_gpu(valid_tensor, angles, hfov, vfov,
                                                          gauge_angles_deg=gauge, mode="nearest")
                valid_patches = ((valid_patches > 0.5) & (n_den > 0.5)).float()
                normals = F.normalize((n_num / n_den.clamp_min(1e-6)).float(), dim=1, eps=1e-6) * valid_patches
            rgb_patches = rgb_patches.detach().cpu()
            views = self.normalize(rgb_patches)
            normals = normals.detach().cpu()
            valid_patches = valid_patches.detach().cpu()
            output = (views, angles.float(), gauge_angles.float(), normals, valid_patches) if self.use_gctt else (views, angles.float(), normals, valid_patches)
            # The validation-only path lets the engine render the untouched official GT.
            return (*output, item["normal"]) if not self.is_train else output
        except Exception as exc:
            raise RuntimeError(f"Failed official RGB/normal pair: {item!r}") from exc


def build_normal_dataset(is_train, args):
    return Stanford2D3DNormalGCTTDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=getattr(args, "input_size", None),
        is_train=is_train,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
