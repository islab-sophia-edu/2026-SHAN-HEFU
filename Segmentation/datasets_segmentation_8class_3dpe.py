import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _strip_known_suffixes(stem: str) -> str:
    """Normalize common RGB/mask suffixes so foo_mask.png can match foo_rgb.jpg."""
    out = stem
    suffixes = (
        "_mask", "-mask", "_masks", "-masks",
        "_anno", "-anno", "_annotation", "-annotation",
        "_label", "-label", "_labels", "-labels",
        "_semantic", "-semantic", "_seg", "-seg",
        "_rgb", "-rgb", "_color", "-color",
    )
    changed = True
    while changed:
        changed = False
        low = out.lower()
        for suf in suffixes:
            if low.endswith(suf):
                out = out[: -len(suf)]
                changed = True
                break
    return out


class ERPRotator:
    """
    3DPE/PanoMAE-style ERP 3D rotation.
    RGB uses bicubic interpolation; masks must use nearest interpolation.
    """
    def __init__(self, h, w, device="cpu"):
        self.h, self.w = h, w
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
        mode: str = "bicubic",
    ) -> torch.Tensor:
        if yaw == 0 and pitch == 0 and roll == 0:
            return img_tensor

        orig_device = img_tensor.device
        img_4d = img_tensor.to(self.device).unsqueeze(0)

        y_, p_, r_ = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor(
            [[np.cos(y_), -np.sin(y_), 0.0],
             [np.sin(y_),  np.cos(y_), 0.0],
             [0.0,         0.0,        1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        Ry = torch.tensor(
            [[ np.cos(p_), 0.0, np.sin(p_)],
             [ 0.0,        1.0, 0.0       ],
             [-np.sin(p_), 0.0, np.cos(p_)]],
            device=self.device,
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [[1.0, 0.0,        0.0       ],
             [0.0, np.cos(r_), -np.sin(r_)],
             [0.0, np.sin(r_),  np.cos(r_)]],
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


class PanoSegmentationDataset(Dataset):
    """
    CVRG-Pano segmentation dataset aligned with 3DPE pretraining.

    Expected directory layout:
        train/rgb/xxx.jpg|png
        train/mask/xxx.png
        test/rgb/xxx.jpg|png
        test/mask/xxx.png

    Returns:
        views:   [N, 3, patch_h, patch_w], ImageNet-normalized tangent patches
        angles:  [N, 2], [lon, lat] degrees
        targets: [N, patch_h, patch_w], segmentation labels

    Training-only augmentations:
        - ERP full 3D pose rotation: yaw [0, 360], pitch/roll [-30, 30]
        - ColorJitter(0.2, 0.2, 0.2, 0.05), p=0.8, RGB only
        - GaussianBlur(kernel_size=7, sigma=(0.1, 1.5)), p=0.1, RGB only
        - optional horizontal roll, applied to RGB and mask
        - multiscale FOV sampling [0.8, 1.2]
        - angle jitter in lon/lat, default +/- 5 degrees

    Evaluation disables all stochastic augmentation and uses the canonical grid.
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

        rgb_folder = getattr(args, "rgb_folder", "rgb")
        mask_folder = getattr(args, "mask_folder", "mask")
        self.rgb_dir = os.path.join(root_dir, rgb_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)

        if args is not None and hasattr(args, "pano_h") and hasattr(args, "pano_w"):
            self.pano_h = int(args.pano_h)
            self.pano_w = int(args.pano_w)
        else:
            raise ValueError("args.pano_h and args.pano_w are required for CVRG-Pano segmentation.")

        self.grid_height = int(grid_height)
        self.v_steps = self.grid_height
        self.u_steps = 2 * self.grid_height
        self.base_v_fov_deg = 180.0 / self.v_steps
        self.base_h_fov_deg = 360.0 / self.u_steps

        if img_size is None:
            self.patch_h = self.pano_h // self.v_steps
            self.patch_w = self.pano_w // self.u_steps
        elif isinstance(img_size, (tuple, list)):
            self.patch_h, self.patch_w = int(img_size[0]), int(img_size[1])
        else:
            self.patch_h = self.patch_w = int(img_size)

        self.base_angles_rad = self._generate_grid_angles_rad()

        self.multiscale_sampling = bool(getattr(args, "multiscale_sampling", True)) and is_train
        self.use_full_pose3d = bool(getattr(args, "use_full_pose3d", True)) and is_train
        self.use_horizontal_roll = bool(getattr(args, "use_horizontal_roll", False)) and is_train
        self.angle_jitter_deg = float(getattr(args, "angle_jitter_deg", 5.0)) if is_train else 0.0

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if bool(getattr(args, "use_color_jitter", True)) and is_train
            else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if bool(getattr(args, "use_blur", True)) and is_train
            else None
        )

        aug_device = getattr(args, "aug_device", "cuda")
        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else "cpu")

        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

        split_name = "Train" if is_train else "Val"
        print(f"[{split_name}] Root       : {self.root_dir}")
        print(f"[{split_name}] RGB Source : {self.rgb_dir}")
        print(f"[{split_name}] Mask Source: {self.mask_dir}")
        print(
            f"[{split_name}] 3DPE augment: "
            f"full_pose3d={self.use_full_pose3d}, multiscale={self.multiscale_sampling}, "
            f"color_jitter={self.color_jitter is not None}, blur={self.blur is not None}, "
            f"angle_jitter_deg={self.angle_jitter_deg}"
        )

        if not os.path.isdir(self.rgb_dir):
            raise ValueError(f"RGB directory not found: {self.rgb_dir}")
        if not os.path.isdir(self.mask_dir):
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        self.file_pairs = self._build_file_pairs(debug_limit=debug_limit)
        print(f"[{split_name}] Successfully paired {len(self.file_pairs)} images.")
        if len(self.file_pairs) == 0:
            rgb_preview = sorted(os.listdir(self.rgb_dir))[:5]
            mask_preview = sorted(os.listdir(self.mask_dir))[:5]
            raise ValueError(
                "No RGB/mask pairs were found. Expected local layout like "
                f"{self.root_dir}/rgb and {self.root_dir}/mask with matching basenames.\n"
                f"RGB examples: {rgb_preview}\n"
                f"Mask examples: {mask_preview}"
            )

    def _build_file_pairs(self, debug_limit=None):
        rgb_files = [f for f in os.listdir(self.rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS]
        mask_files = [f for f in os.listdir(self.mask_dir) if os.path.splitext(f)[1] in MASK_EXTS]
        rgb_files = sorted(rgb_files)
        mask_files = sorted(mask_files)
        if debug_limit:
            mask_files = mask_files[:debug_limit]

        rgb_index = {}
        for rgb_f in rgb_files:
            stem = os.path.splitext(rgb_f)[0]
            keys = {
                stem.lower(),
                _strip_known_suffixes(stem).lower(),
            }
            for k in keys:
                rgb_index.setdefault(k, rgb_f)

        pairs = []
        missing = []
        for mask_f in mask_files:
            mask_stem = os.path.splitext(mask_f)[0]
            candidate_keys = [
                mask_stem.lower(),
                _strip_known_suffixes(mask_stem).lower(),
            ]

            rgb_f = None
            for key in candidate_keys:
                if key in rgb_index:
                    rgb_f = rgb_index[key]
                    break

            if rgb_f is None:
                missing.append(mask_f)
                continue

            pairs.append((
                os.path.join(self.mask_dir, mask_f),
                os.path.join(self.rgb_dir, rgb_f),
            ))

        if missing:
            print(f"Warning: {len(missing)} masks have no corresponding RGB image in '{self.rgb_dir}'.")
            print(f"Missing examples: {missing[:5]}")
        return pairs

    def _generate_grid_angles_rad(self):
        phis = np.linspace(
            np.pi / 2.0 - np.radians(self.base_v_fov_deg) / 2.0,
            -np.pi / 2.0 + np.radians(self.base_v_fov_deg) / 2.0,
            self.v_steps,
        )
        thetas = np.linspace(-np.pi, np.pi, self.u_steps, endpoint=False)
        grid_phis, grid_thetas = np.meshgrid(phis, thetas, indexing="ij")
        return np.stack([grid_thetas.flatten(), grid_phis.flatten()], axis=1)

    def _get_rotator(self, h, w):
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h, self._rotator_w = h, w
        return self._rotator

    def _extract_patches_tensor(
        self,
        pano_tensor: torch.Tensor,
        angles_deg,
        h_fov_rad: float,
        v_fov_rad: float,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """
        Extract tangent/perspective patches using the same spherical geometry as 3DPE.
        Supports RGB [3,H,W] and mask [1,H,W] tensors.
        """
        device = self.aug_device
        N = len(angles_deg)
        P_h, P_w = self.patch_h, self.patch_w

        angles_t = torch.tensor(angles_deg, dtype=torch.float32, device=device)
        theta = torch.deg2rad(angles_t[:, 0])
        phi = torch.deg2rad(angles_t[:, 1])

        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        nc = torch.stack(
            [cos_phi * cos_theta, -cos_phi * sin_theta, sin_phi],
            dim=1,
        )
        xn = torch.stack(
            [-sin_theta, -cos_theta, torch.zeros_like(theta)],
            dim=1,
        )
        yn = torch.stack(
            [-sin_phi * cos_theta, sin_phi * sin_theta, cos_phi],
            dim=1,
        )

        u_lin = torch.linspace(-1.0, 1.0, P_w, device=device)
        v_lin = torch.linspace(1.0, -1.0, P_h, device=device)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing="xy")
        plane_x = torch.tan(torch.tensor(h_fov_rad / 2.0, device=device)) * uu.reshape(-1)
        plane_y = torch.tan(torch.tensor(v_fov_rad / 2.0, device=device)) * vv.reshape(-1)

        pts = (
            nc.unsqueeze(1)
            + plane_x.view(1, -1, 1) * xn.unsqueeze(1)
            + plane_y.view(1, -1, 1) * yn.unsqueeze(1)
        )
        pts = F.normalize(pts, dim=-1)

        px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
        theta_odi = torch.atan2(-py, px)
        phi_odi = torch.asin(torch.clamp(pz, -1.0, 1.0))

        grid_x = theta_odi / torch.pi
        grid_y = -phi_odi / (torch.pi / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, P_h, P_w, 2)

        pano_batch = pano_tensor.to(device).unsqueeze(0).expand(N, -1, -1, -1)
        patches = F.grid_sample(
            pano_batch,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )
        return patches

    def _load_and_process_pair(self, idx):
        mask_path, img_path = self.file_pairs[idx]

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)

        if img_pil.size != (self.pano_w, self.pano_h):
            img_pil = img_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)
        if mask_pil.size != (self.pano_w, self.pano_h):
            mask_pil = mask_pil.resize((self.pano_w, self.pano_h), Image.NEAREST)

        if self.color_jitter is not None and np.random.rand() < 0.8:
            img_pil = self.color_jitter(img_pil)
        if self.blur is not None and np.random.rand() < 0.1:
            img_pil = self.blur(img_pil)

        img_np = np.array(img_pil)
        mask_np = np.array(mask_pil)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        roll_shift_deg = 0.0
        if self.use_horizontal_roll:
            shift = np.random.randint(0, self.pano_w)
            img_np = np.roll(img_np, shift, axis=1)
            mask_np = np.roll(mask_np, shift, axis=1)
            roll_shift_deg = (shift / self.pano_w) * 360.0

        img_tensor = torch.from_numpy(img_np.copy()).float().div(255.0).permute(2, 0, 1).to(self.aug_device)
        mask_tensor = torch.from_numpy(mask_np.copy().astype(np.float32)).unsqueeze(0).to(self.aug_device)

        if self.use_full_pose3d:
            yaw_shift = np.random.uniform(0.0, 360.0)
            pitch_shift = np.random.uniform(-30.0, 30.0)
            roll_shift = np.random.uniform(-30.0, 30.0)
            rotator = self._get_rotator(self.pano_h, self.pano_w)

            img_tensor = rotator.rotate_tensor(
                img_tensor, yaw_shift, pitch_shift, roll_shift, mode="bicubic"
            )
            mask_tensor = rotator.rotate_tensor(
                mask_tensor, yaw_shift, pitch_shift, roll_shift, mode="nearest"
            )

        return img_tensor, mask_tensor, roll_shift_deg, mask_path, img_path

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        try:
            img_tensor, mask_tensor, roll_shift_deg, _, _ = self._load_and_process_pair(idx)

            scale = np.random.uniform(0.8, 1.2) if self.multiscale_sampling else 1.0
            curr_h_fov_rad = np.radians(self.base_h_fov_deg * scale)
            curr_v_fov_rad = np.radians(self.base_v_fov_deg * scale)

            angles_list = []
            for base_theta, base_phi in self.base_angles_rad:
                jitter_t = np.radians(
                    np.random.uniform(-self.angle_jitter_deg, self.angle_jitter_deg)
                )
                jitter_p = np.radians(
                    np.random.uniform(-self.angle_jitter_deg, self.angle_jitter_deg)
                )
                roll_rad = np.radians(roll_shift_deg)
                curr_theta = base_theta + jitter_t + roll_rad
                curr_phi = np.clip(
                    base_phi + jitter_p,
                    -np.pi / 2.0 + curr_v_fov_rad / 2.0,
                    np.pi / 2.0 - curr_v_fov_rad / 2.0,
                )
                angles_list.append([np.degrees(curr_theta), np.degrees(curr_phi)])

            rgb_patches = self._extract_patches_tensor(
                img_tensor,
                angles_list,
                curr_h_fov_rad,
                curr_v_fov_rad,
                mode="bicubic",
            )
            mask_patches = self._extract_patches_tensor(
                mask_tensor,
                angles_list,
                curr_h_fov_rad,
                curr_v_fov_rad,
                mode="nearest",
            ).squeeze(1).long()

            mean = self._norm_mean.to(self.aug_device)
            std = self._norm_std.to(self.aug_device)
            rgb_patches = (rgb_patches - mean) / std

            views = rgb_patches.cpu()
            angles = torch.tensor(angles_list, dtype=torch.float32)
            targets = mask_patches.cpu()

            return views, angles, targets

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            import traceback
            traceback.print_exc()

            dummy_n = self.u_steps * self.v_steps
            angles = torch.tensor(
                [[np.degrees(t), np.degrees(p)] for t, p in self.base_angles_rad],
                dtype=torch.float32,
            )
            return (
                torch.zeros(dummy_n, 3, self.patch_h, self.patch_w),
                angles,
                torch.full((dummy_n, self.patch_h, self.patch_w), 255, dtype=torch.long),
            )


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
