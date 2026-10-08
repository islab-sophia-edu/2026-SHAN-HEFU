# Noadaptive variant: segmentation uses full token grid; no MAE adaptive masking is applied.
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ERPRotator:
    """
    Rotation augmentation for ERP panoramas.

    RGB uses bicubic interpolation; semantic masks must use nearest interpolation
    to avoid corrupting class ids. Rotation is used only when is_train=True.
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
        self.theta = u * np.pi
        self.phi = v * np.pi / 2.0
        self.xyz = torch.stack([
            torch.cos(self.phi) * torch.cos(self.theta),
            -torch.cos(self.phi) * torch.sin(self.theta),
            torch.sin(self.phi),
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
            [1.0, 0.0,        0.0],
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
    CVRG-Pano semantic segmentation dataset adapted to the GCTT data contract.

    Returns:
        views, angles, gauge_angles, masks

    Shapes before DataLoader batching:
        views        : [N, 3, patch_h, patch_w]
        angles       : [N, 2], degrees, ordered as [lon/theta, lat/phi]
        gauge_angles : [N, 1], degrees, tangent-frame in-plane gauge psi_i
        masks        : [N, patch_h, patch_w], int64 labels

    Important:
        The RGB patch and mask patch are extracted with the same tangent frame
        and the same gauge angle. Therefore the segmentation target remains
        aligned with the input view even when GCTT gauge jitter is enabled.
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
        self.is_train = is_train
        self.num_classes = num_classes
        self.args = args

        self.mask_dir = os.path.join(root_dir, "mask")

        # CVRG-Pano split layout used by the user:
        #   train/rgb, train/mask, test/rgb, test/mask
        # If --rgb_shared_dir is not explicitly provided, use root_dir/rgb.
        rgb_arg = getattr(args, "rgb_shared_dir", None) if args is not None else None
        self.rgb_shared_dir = rgb_arg if rgb_arg else os.path.join(root_dir, "rgb")

        self.v_steps = int(grid_height)
        self.u_steps = 2 * int(grid_height)
        self.v_fov = 180.0 / self.v_steps
        self.h_fov = 360.0 / self.u_steps

        # Resolve panorama and patch sizes.
        self.target_h = None
        self.target_w = None
        if args is not None and hasattr(args, "pano_h") and hasattr(args, "pano_w"):
            self.target_h = int(args.pano_h)
            self.target_w = int(args.pano_w)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            # img_size in this segmentation code is usually patch size.
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

        print(f"[{'Train' if is_train else 'Val'}] Mask Source: {self.mask_dir}")
        print(f"[{'Train' if is_train else 'Val'}] RGB Source : {self.rgb_shared_dir}")
        print(f"[{'Train' if is_train else 'Val'}] Patch Size : {self.patch_h}x{self.patch_w}")

        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Annotation directory not found: {self.mask_dir}")

        # Do not fail here: the matcher below tries several possible RGB directories.

        valid_mask_exts = {".png"}
        valid_rgb_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        mask_files = sorted([
            f for f in os.listdir(self.mask_dir)
            if os.path.splitext(f)[1].lower() in valid_mask_exts
        ])
        if debug_limit:
            mask_files = mask_files[:debug_limit]

        def _list_images_recursive(base_dir):
            """Return image file paths relative to base_dir. Supports nested RGB dirs."""
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
            """
            CVRG-Pano copies sometimes store RGB in all-rgb, rgb, image/images,
            or inside the train/test split. Try likely locations before failing.
            """
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

        import re

        DROP_TOKENS = {
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
                # Remove common semantic-mask/RGB role words, but preserve IDs.
                if t in DROP_TOKENS:
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
            """
            Aggressive but deterministic filename keys.
            Handles examples like:
                xxx.png <-> xxx.jpg
                xxx_mask.png <-> xxx.jpg
                mask_xxx.png <-> rgb_xxx.jpg
                xxx_labelIds.png <-> xxx_rgb.jpg
                000123_8class.png <-> 000123.jpg
            """
            stem = os.path.splitext(os.path.basename(filename))[0].lower()
            tokens = _split_tokens(filename)
            stripped = _strip_tokens(tokens)
            keys = {stem, "".join(tokens), "_".join(tokens)}
            if stripped:
                keys.add("".join(stripped))
                keys.add("_".join(stripped))
            # Normalize numbers with and without leading zeros.
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
                # Prefer non-numeric semantic keys; numeric keys are fallback.
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
                "No RGB/mask pairs were found. The dataset is empty, so RandomSampler "
                "would fail with num_samples=0. Fix --rgb_shared_dir or update the "
                "filename matcher. The log above prints the tried RGB dirs, example "
                "missing masks, and example RGB files."
            )

        v_centers = torch.linspace(90.0 - self.v_fov / 2.0, -90.0 + self.v_fov / 2.0, self.v_steps)
        u_centers = torch.linspace(-180.0, 180.0, self.u_steps + 1)[:-1]
        v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing="ij")
        self.angle_centers = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1).float()

        self.use_gctt = bool(getattr(args, "use_gctt", True))
        self.gctt_gauge_jitter_deg = float(getattr(args, "gctt_gauge_jitter_deg", 30.0)) if is_train else 0.0
        self.gctt_local_gauge_jitter_deg = float(getattr(args, "gctt_local_gauge_jitter_deg", 0.0)) if is_train else 0.0
        self.angle_jitter_deg = float(getattr(args, "angle_jitter_deg", 0.0)) if is_train else 0.0

        aug_device = getattr(args, "aug_device", "cuda")
        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else "cpu")

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if bool(getattr(args, "use_color_jitter", True)) and is_train else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if bool(getattr(args, "use_blur", True)) and is_train else None
        )

        # Pretraining-style full 3D pose augmentation. Disabled automatically for val/test.
        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", True)) and is_train
        self.pose_yaw_deg = float(getattr(args, "pose_yaw_deg", 360.0))
        self.pose_pitch_deg = float(getattr(args, "pose_pitch_deg", 30.0))
        self.pose_roll_deg = float(getattr(args, "pose_roll_deg", 30.0))
        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Pixel coordinates in each tangent patch. Non-square patches are supported.
        u_lin = torch.linspace(-(self.patch_w - 1) / 2.0, (self.patch_w - 1) / 2.0, self.patch_w)
        v_lin = torch.linspace((self.patch_h - 1) / 2.0, -(self.patch_h - 1) / 2.0, self.patch_h)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing="xy")
        self._pixel_u = uu.reshape(-1)
        self._pixel_v = vv.reshape(-1)

    def _get_rotator(self, h, w):
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h = h
            self._rotator_w = w
        return self._rotator

    def _apply_full_pose3d(self, img_np: np.ndarray, mask_np: np.ndarray):
        """Apply the same ERP 3D rotation to RGB and mask. Train only."""
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

    @staticmethod
    def _to_tensor_rgb(img_pil: Image.Image) -> torch.Tensor:
        return transforms.ToTensor()(img_pil)

    def _make_angles_and_gauges(self):
        angles = self.angle_centers.clone()

        if self.angle_jitter_deg > 0.0:
            jitter = torch.empty_like(angles).uniform_(-self.angle_jitter_deg, self.angle_jitter_deg)
            angles = angles + jitter
            # Keep latitude valid for the current vertical FoV.
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
        gauge_angles_deg: torch.Tensor = None,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """
        Extract tangent-plane patches with an explicit GCTT local gauge.

        The local frame is:
            nc: patch center direction
            xn: canonical local right axis
            yn: canonical local up axis

        GCTT rotates the tangent frame:
            x_g = cos(psi) * xn + sin(psi) * yn
            y_g = -sin(psi) * xn + cos(psi) * yn
        """
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

        nc = torch.stack([
            cos_phi * cos_theta,
            -cos_phi * sin_theta,
            sin_phi,
        ], dim=1)

        xn = torch.stack([
            -sin_theta,
            -cos_theta,
            torch.zeros_like(theta),
        ], dim=1)

        yn = torch.stack([
            -sin_phi * cos_theta,
            sin_phi * sin_theta,
            cos_phi,
        ], dim=1)

        if gauge_angles_deg is None:
            gauge = torch.zeros(N, device=device, dtype=torch.float32)
        else:
            gauge = gauge_angles_deg.to(device=device, dtype=torch.float32).view(N)

        psi = torch.deg2rad(gauge)
        c = torch.cos(psi).unsqueeze(1)
        s = torch.sin(psi).unsqueeze(1)
        xg = c * xn + s * yn
        yg = -s * xn + c * yn

        # General pinhole projection. For square tangent patches this reduces to
        # the same geometry as the pretraining code, but it also works for H != W.
        fx = (self.patch_w / 2.0) / torch.tan(torch.tensor(h_fov_rad / 2.0, device=device))
        fy = (self.patch_h / 2.0) / torch.tan(torch.tensor(v_fov_rad / 2.0, device=device))

        uu = self._pixel_u.to(device)
        vv = self._pixel_v.to(device)
        pts = (
            (uu.view(1, -1, 1) / fx) * xg.unsqueeze(1)
            + (vv.view(1, -1, 1) / fy) * yg.unsqueeze(1)
            + nc.unsqueeze(1)
        )

        px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
        p_norm = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-8)

        theta_erp = torch.atan2(-py, px)
        phi_erp = torch.asin(torch.clamp(pz / p_norm, -1.0, 1.0))

        grid_x = theta_erp / torch.pi
        grid_y = -phi_erp / (torch.pi / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, self.patch_h, self.patch_w, 2)

        pano_batch = pano_tensor.unsqueeze(0).expand(N, -1, -1, -1)
        return F.grid_sample(
            pano_batch,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )

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

            # Pretraining-style 3D pose augmentation. RGB uses bicubic; mask uses nearest.
            # Evaluation/test never enters this path because self.use_full_pose3d=False.
            img_np, mask_np = self._apply_full_pose3d(img_np, mask_np)

            # Horizontal panorama roll is safe because RGB and mask are rolled together.
            if self.is_train and bool(getattr(self.args, "use_horizontal_roll", True)):
                roll_idx = random.randint(0, img_np.shape[1] - 1)
                img_np = np.roll(img_np, roll_idx, axis=1).copy()
                mask_np = np.roll(mask_np, roll_idx, axis=1).copy()

            angles, gauge_angles = self._make_angles_and_gauges()

            rgb_tensor = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

            rgb_patches = self._extract_patches_gpu(
                rgb_tensor,
                angles,
                np.radians(self.h_fov),
                np.radians(self.v_fov),
                gauge_angles_deg=gauge_angles if self.use_gctt else None,
                mode="bicubic",
            )
            mask_patches = self._extract_patches_gpu(
                mask_tensor,
                angles,
                np.radians(self.h_fov),
                np.radians(self.v_fov),
                gauge_angles_deg=gauge_angles if self.use_gctt else None,
                mode="nearest",
            )

            # Bring tensors back to CPU for DataLoader collation. This avoids
            # keeping CUDA tensors alive across worker boundaries.
            rgb_patches = rgb_patches.detach().cpu()
            mask_patches = mask_patches[:, 0].round().long().detach().cpu()

            rgb_patches = torch.stack([self.normalize(p) for p in rgb_patches], dim=0)

        except Exception as e:
            print(f"Error loading {mask_name} / {rgb_name}: {e}")
            n = self.u_steps * self.v_steps
            return (
                torch.zeros(n, 3, self.patch_h, self.patch_w),
                self.angle_centers.clone(),
                torch.zeros(n, 1),
                torch.zeros(n, self.patch_h, self.patch_w).long(),
            )

        return rgb_patches, angles, gauge_angles, mask_patches


def build_segmentation_dataset(is_train, args):
    dataset = PanoSegmentationDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=args.input_size,
        is_train=is_train,
        num_classes=args.nb_classes,
        debug_limit=getattr(args, "debug_limit", None),
        args=args,
    )
    return dataset
