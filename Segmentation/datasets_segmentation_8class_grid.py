import os
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ERPRotator:
    """
    Full-ERP rotation augmentation.

    RGB uses bicubic interpolation; semantic masks use nearest interpolation so
    class ids are not corrupted. This is an ERP-level augmentation only. It does
    not perform ODI/tangent-plane tokenization.
    """

    def __init__(self, h, w, device="cpu"):
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
        self.xyz = torch.stack([
            torch.cos(phi) * torch.cos(theta),
            -torch.cos(phi) * torch.sin(theta),
            torch.sin(phi),
        ], dim=-1).view(-1, 3)

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
        Rz = torch.tensor([
            [np.cos(y_), -np.sin(y_), 0.0],
            [np.sin(y_),  np.cos(y_), 0.0],
            [0.0,        0.0,        1.0],
        ], device=self.device, dtype=torch.float32)
        Ry = torch.tensor([
            [ np.cos(p_), 0.0, np.sin(p_)],
            [ 0.0,        1.0, 0.0],
            [-np.sin(p_), 0.0, np.cos(p_)],
        ], device=self.device, dtype=torch.float32)
        Rx = torch.tensor([
            [1.0, 0.0,         0.0],
            [0.0, np.cos(r_), -np.sin(r_)],
            [0.0, np.sin(r_),  np.cos(r_)],
        ], device=self.device, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([
            torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi,
            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1.0, 1.0)) / (np.pi / 2.0)),
        ], dim=-1).view(1, self.h, self.w, 2)

        rotated = F.grid_sample(
            img_4d,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )
        return rotated.squeeze(0).to(orig_device)


class PanoSegmentationDataset(Dataset):
    """
    CVRG-Pano semantic segmentation dataset for the grid-token ablation.

    Tokenization:
        Pure ERP grid patchify, MAE-style row-major patch sequence.
        No ODI, no tangent-plane projection, no FoV sampling, no overlap.

    Returns before DataLoader batching:
        GCTT-grid mode:
            views, angles, gauge_angles, masks
        PB/center-only mode (--no_gctt):
            views, angles, masks

    Shapes:
        views        : [N, 3, patch_h, patch_w]
        angles       : [N, 2], degrees, ordered as [lon/theta, lat/phi]
        gauge_angles : [N, 1], all zeros in grid ablation
        masks        : [N, patch_h, patch_w], int64 labels
    """

    def __init__(
        self,
        root_dir,
        grid_height=16,
        img_size=None,
        is_train=True,
        num_classes=8,
        debug_limit=None,
        args=None,
    ):
        self.root_dir = root_dir
        self.is_train = bool(is_train)
        self.num_classes = int(num_classes)
        self.args = args

        self.mask_dir = os.path.join(root_dir, "mask")
        rgb_arg = getattr(args, "rgb_shared_dir", None) if args is not None else None
        self.rgb_shared_dir = rgb_arg if rgb_arg else os.path.join(root_dir, "rgb")

        self.v_steps = int(grid_height)
        self.u_steps = 2 * int(grid_height)
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps

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

        print(f"[{'Train' if is_train else 'Val'}] CVRG-Pano GRID Segmentation")
        print(f"[{'Train' if is_train else 'Val'}] Mask Source: {self.mask_dir}")
        print(f"[{'Train' if is_train else 'Val'}] RGB Source : {self.rgb_shared_dir}")
        print(f"[{'Train' if is_train else 'Val'}] Patch Size : {self.patch_h}x{self.patch_w}")
        print(f"[{'Train' if is_train else 'Val'}] Tokenizer  : ERP grid patchify; no ODI / no tangent-plane")

        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Annotation directory not found: {self.mask_dir}")

        valid_mask_exts = {".png"}
        valid_rgb_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        mask_files = sorted([
            f for f in os.listdir(self.mask_dir)
            if os.path.splitext(f)[1].lower() in valid_mask_exts
        ])
        if debug_limit:
            mask_files = mask_files[:debug_limit]

        def _list_images_recursive(base_dir):
            out = []
            if not os.path.isdir(base_dir):
                return out
            for cur, _, files in os.walk(base_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_rgb_exts:
                        rel = os.path.relpath(os.path.join(cur, f), base_dir)
                        out.append(rel)
            return sorted(out)

        def _candidate_rgb_dirs():
            parent = os.path.dirname(os.path.abspath(self.root_dir))
            candidates = []
            if self.rgb_shared_dir:
                candidates.append(self.rgb_shared_dir)
            for d in [
                os.path.join(self.root_dir, "rgb"),
                os.path.join(self.root_dir, "RGB"),
                os.path.join(self.root_dir, "image"),
                os.path.join(self.root_dir, "images"),
                os.path.join(self.root_dir, "img"),
                os.path.join(parent, "all-rgb"),
                os.path.join(parent, "all_rgb"),
                os.path.join(parent, "rgb"),
                os.path.join(parent, "RGB"),
                os.path.join(parent, "image"),
                os.path.join(parent, "images"),
                os.path.join(parent, "img"),
            ]:
                candidates.append(d)
            seen = set()
            unique = []
            for d in candidates:
                d = os.path.abspath(d)
                if d not in seen:
                    seen.add(d)
                    unique.append(d)
            return unique

        drop_tokens = {
            "mask", "masks", "anno", "annotation", "annotations",
            "label", "labels", "labelid", "labelids", "gt", "gtfine",
            "seg", "semantic", "segmentation", "rgb", "image", "img",
            "color", "colour", "pano", "panorama", "class", "classes",
            "8class", "8classes",
        }

        def _split_tokens(filename):
            stem = os.path.splitext(os.path.basename(filename))[0].lower()
            return [t for t in re.split(r"[^a-z0-9]+", stem) if t]

        def _strip_tokens(tokens):
            stripped = []
            for t in tokens:
                if t in drop_tokens:
                    continue
                if t.endswith("mask") and len(t) > 4:
                    t = t[:-4]
                if t.endswith("label") and len(t) > 5:
                    t = t[:-5]
                if t.endswith("labelids") and len(t) > 8:
                    t = t[:-8]
                if t.startswith("mask") and len(t) > 4:
                    t = t[4:]
                if t.startswith("rgb") and len(t) > 3:
                    t = t[3:]
                if t:
                    stripped.append(t)
            return stripped

        def _stem_keys(filename):
            stem = os.path.splitext(os.path.basename(filename))[0].lower()
            tokens = _split_tokens(filename)
            stripped = _strip_tokens(tokens)
            keys = {stem, "".join(tokens), "_".join(tokens)}
            if stripped:
                keys.add("".join(stripped))
                keys.add("_".join(stripped))
            nums = re.findall(r"\d+", stem)
            if nums:
                keys.add("#all:" + "_".join(nums))
                keys.add("#first:" + nums[0].lstrip("0"))
                keys.add("#first_raw:" + nums[0])
                keys.add("#last:" + nums[-1].lstrip("0"))
                keys.add("#last_raw:" + nums[-1])
                if len(nums) >= 2:
                    keys.add("#first2:" + "_".join(n.lstrip("0") for n in nums[:2]))
                    keys.add("#last2:" + "_".join(n.lstrip("0") for n in nums[-2:]))
            return {k for k in keys if k and k not in {"#first:", "#last:"}}

        def _build_rgb_index(rgb_files):
            rgb_index = {}
            ambiguous = set()
            for rgb_f in rgb_files:
                for key in _stem_keys(rgb_f):
                    if key in rgb_index and rgb_index[key] != rgb_f:
                        ambiguous.add(key)
                    else:
                        rgb_index[key] = rgb_f
            for key in ambiguous:
                rgb_index.pop(key, None)
            return rgb_index

        def _pair_with_rgb_dir(rgb_dir):
            rgb_files = _list_images_recursive(rgb_dir)
            rgb_index = _build_rgb_index(rgb_files)
            pairs, missing = [], []
            for mask_f in mask_files:
                found_rgb = None
                keys = _stem_keys(mask_f)
                for key in sorted(keys, key=lambda x: (x.startswith("#"), len(x))):
                    if key in rgb_index:
                        found_rgb = rgb_index[key]
                        break
                if found_rgb is not None:
                    pairs.append((mask_f, found_rgb))
                else:
                    missing.append(mask_f)
            return pairs, missing, rgb_files

        print(f"[{'Train' if is_train else 'Val'}] Matching RGB files for {len(mask_files)} masks...")
        self.file_pairs = []
        missing_masks = mask_files
        chosen_rgb_files = []
        chosen_rgb_dir = self.rgb_shared_dir

        for cand_dir in _candidate_rgb_dirs():
            pairs, missing, rgb_files = _pair_with_rgb_dir(cand_dir)
            print(
                f"[{'Train' if is_train else 'Val'}] Try RGB dir: {cand_dir} | "
                f"RGB files: {len(rgb_files)} | paired: {len(pairs)}"
            )
            if len(pairs) > len(self.file_pairs):
                self.file_pairs = pairs
                missing_masks = missing
                chosen_rgb_files = rgb_files
                chosen_rgb_dir = cand_dir
            if len(pairs) == len(mask_files):
                break

        self.rgb_shared_dir = chosen_rgb_dir
        print(f"[{'Train' if is_train else 'Val'}] Selected RGB Source: {self.rgb_shared_dir}")
        print(f"[{'Train' if is_train else 'Val'}] Successfully paired {len(self.file_pairs)} images.")

        if missing_masks:
            print(
                f"Warning: {len(missing_masks)} masks have no corresponding RGB image in "
                f"'{self.rgb_shared_dir}'."
            )
            print(f"Example missing masks: {missing_masks[:10]}")
            print(f"Example RGB files: {chosen_rgb_files[:10]}")

        if len(self.file_pairs) == 0:
            raise ValueError(
                "No RGB/mask pairs were found. Fix --rgb_shared_dir or update the filename matcher."
            )

        v_centers = torch.linspace(90.0 - self.v_fov / 2.0, -90.0 + self.v_fov / 2.0, self.v_steps)
        u_centers = torch.linspace(-180.0, 180.0, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing="ij")
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).float()

        self.use_gctt = bool(getattr(args, "use_gctt", True))
        # In grid ablation there is no local tangent frame to rotate. Gauge jitter
        # is intentionally disabled even if old command lines pass non-zero values.
        self.gctt_gauge_jitter_deg = 0.0
        self.gctt_local_gauge_jitter_deg = 0.0
        self.angle_jitter_deg = 0.0

        aug_device = getattr(args, "aug_device", "cuda") if args is not None else "cuda"
        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else "cpu")

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if bool(getattr(args, "use_color_jitter", True)) and is_train else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if bool(getattr(args, "use_blur", True)) and is_train else None
        )

        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", True)) and is_train
        self.pose_yaw_deg = float(getattr(args, "pose_yaw_deg", 360.0)) if args is not None else 360.0
        self.pose_pitch_deg = float(getattr(args, "pose_pitch_deg", 30.0)) if args is not None else 30.0
        self.pose_roll_deg = float(getattr(args, "pose_roll_deg", 30.0)) if args is not None else 30.0
        self.use_horizontal_roll = bool(getattr(args, "use_horizontal_roll", True)) and is_train
        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _get_rotator(self, h, w):
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
        return len(self.file_pairs)

    def _make_angles_and_gauges(self):
        angles = self.angle_centers.clone().float()
        gauge_angles = torch.zeros(angles.shape[0], 1, dtype=torch.float32)
        return angles, gauge_angles

    def _patchify_rgb_grid(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        c, h, w = rgb_tensor.shape
        if h != self.v_steps * self.patch_h or w != self.u_steps * self.patch_w:
            raise ValueError(
                f"Unexpected RGB tensor size {h}x{w}; expected "
                f"{self.v_steps * self.patch_h}x{self.u_steps * self.patch_w}."
            )
        patches = rgb_tensor.view(c, self.v_steps, self.patch_h, self.u_steps, self.patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()
        return patches.view(self.v_steps * self.u_steps, c, self.patch_h, self.patch_w)

    def _patchify_mask_grid(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        h, w = mask_tensor.shape
        if h != self.v_steps * self.patch_h or w != self.u_steps * self.patch_w:
            raise ValueError(
                f"Unexpected mask tensor size {h}x{w}; expected "
                f"{self.v_steps * self.patch_h}x{self.u_steps * self.patch_w}."
            )
        patches = mask_tensor.view(self.v_steps, self.patch_h, self.u_steps, self.patch_w)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.view(self.v_steps * self.u_steps, self.patch_h, self.patch_w)

    def __getitem__(self, idx):
        mask_name, rgb_name = self.file_pairs[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        img_path = os.path.join(self.rgb_shared_dir, rgb_name)

        try:
            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path)

            if self.target_h is not None and self.target_w is not None:
                if img_pil.size != (self.target_w, self.target_h):
                    img_pil = img_pil.resize((self.target_w, self.target_h), Image.BICUBIC)
                    mask_pil = mask_pil.resize((self.target_w, self.target_h), Image.NEAREST)

            if self.color_jitter is not None and random.random() < 0.8:
                img_pil = self.color_jitter(img_pil)
            if self.blur is not None and random.random() < 0.1:
                img_pil = self.blur(img_pil)

            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]

            img_np, mask_np = self._apply_full_pose3d(img_np, mask_np)

            if self.use_horizontal_roll:
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1).copy()
                mask_np = np.roll(mask_np, roll_idx, axis=1).copy()

            angles, gauge_angles = self._make_angles_and_gauges()

            rgb_tensor = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_np.astype(np.int64)).long()

            rgb_patches = self._patchify_rgb_grid(rgb_tensor)
            rgb_patches = torch.stack([self.normalize(p) for p in rgb_patches], dim=0)
            mask_patches = self._patchify_mask_grid(mask_tensor)

        except Exception as e:
            print(f"Error loading {mask_name} / {rgb_name}: {e}")
            n = self.u_steps * self.v_steps
            rgb_patches = torch.zeros(n, 3, self.patch_h, self.patch_w)
            angles = self.angle_centers.clone()
            gauge_angles = torch.zeros(n, 1)
            mask_patches = torch.zeros(n, self.patch_h, self.patch_w).long()

        if self.use_gctt:
            return rgb_patches, angles, gauge_angles, mask_patches
        return rgb_patches, angles, mask_patches


def build_segmentation_dataset(is_train, args):
    return PanoSegmentationDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=args.input_size,
        is_train=is_train,
        num_classes=args.nb_classes,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
