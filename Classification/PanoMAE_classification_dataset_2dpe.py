import os
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
    ImageFolder with two classification-specific rules:
      1. Exclude the 'others' class.
      2. Optionally enforce the class_to_idx mapping from the training split.
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
    Differentiation-free ERP rotation used as panorama-level data augmentation.

    Coordinate convention is kept consistent with the PanoMAE/GCTT code:
        p(theta, phi) = [cos(phi) cos(theta), -cos(phi) sin(theta), sin(phi)].
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


class PanoClassificationDataset(Dataset):
    """
    2DPE-ablation PanoMAE classification dataset.

    Returns:
        views:  [N, 3, S, S], normalized tangent-view patches.
        angles: [N, 2], degrees, [lon/theta, lat/phi].
        label:  scalar long tensor.

    This version ports the panorama-level augmentations from the pretraining
    dataset: ERP full-pose rotation, horizontal roll, multiscale FoV sampling,
    angle jitter, color jitter, and blur. It intentionally does not return
    gauge_angles because the 2DPE-ablation downstream model keeps the same interface and expects:
        model(views, angles)
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
        # Image / geometry augmentation, aligned with the third pretraining dataset
        use_color_jitter: bool = True,
        use_blur: bool = True,
        use_full_pose3d: bool = True,
        use_horizontal_roll: bool = False,
        multiscale_sampling: bool = True,
        angle_jitter_deg: float = 5.0,
        # Runtime
        aug_device: str = "cpu",
        return_device: str = "cpu",
        reprob: float = 0.0,
    ):
        self.root_dir = root_dir
        self.image_folder = StrictImageFolder(
            root=root_dir,
            predefined_class_to_idx=class_to_idx,
        )
        self.class_to_idx = self.image_folder.class_to_idx
        self.classes = self.image_folder.classes
        self.is_train = bool(is_train)

        mode = "Train" if is_train else "Val"
        print(f"[{mode}] Loaded {len(self.classes)} classes after filtering 'others'.")
        print(f"[{mode}] Number of images: {len(self.image_folder.samples)}")

        self.pano_h = int(pano_h)
        self.pano_w = int(pano_w)
        self.grid_height = int(grid_height)
        self.v_num = self.grid_height
        self.u_num = 2 * self.grid_height
        self.patch_size = int(img_size)

        self.base_v_fov_deg = 180.0 / self.v_num
        self.base_h_fov_deg = 360.0 / self.u_num
        self.base_angles_rad = self._generate_grid_angles_rad()

        self.multiscale_sampling = bool(multiscale_sampling) and self.is_train
        self.use_full_pose3d = bool(use_full_pose3d) and self.is_train
        self.use_horizontal_roll = bool(use_horizontal_roll) and self.is_train
        self.angle_jitter_deg = float(angle_jitter_deg) if self.is_train else 0.0

        # Apply appearance transforms at panorama level, before tangent extraction.
        # Do not use RandomHorizontalFlip on individual tangent patches: it changes
        # the local tangent orientation without any corresponding positional state.
        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if (use_color_jitter and self.is_train)
            else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if (use_blur and self.is_train)
            else None
        )

        self.reprob = float(reprob) if self.is_train else 0.0

        self.aug_device = torch.device(
            aug_device if (str(aug_device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.return_device = torch.device(return_device)

        P = self.patch_size
        u_lin = torch.linspace(-(P - 1) / 2.0, (P - 1) / 2.0, P)
        v_lin = torch.linspace((P - 1) / 2.0, -(P - 1) / 2.0, P)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing="xy")
        self._pixel_u = uu.reshape(-1)
        self._pixel_v = vv.reshape(-1)

        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

    def _generate_grid_angles_rad(self) -> np.ndarray:
        phis = np.linspace(
            np.pi / 2.0 - np.radians(self.base_v_fov_deg) / 2.0,
            -np.pi / 2.0 + np.radians(self.base_v_fov_deg) / 2.0,
            self.v_num,
        )
        thetas = np.linspace(-np.pi, np.pi, self.u_num, endpoint=False)
        grid_phis, grid_thetas = np.meshgrid(phis, thetas, indexing="ij")
        return np.stack([grid_thetas.flatten(), grid_phis.flatten()], axis=1)

    def _get_rotator(self, h: int, w: int) -> ERPRotator:
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h = h
            self._rotator_w = w
        return self._rotator

    def _extract_patches_no_gctt(
        self,
        pano_tensor: torch.Tensor,
        angles_deg,
        h_fov_rad: float,
        v_fov_rad: float,
    ) -> torch.Tensor:
        """
        Extract tangent patches with the canonical local tangent frame.

        Canonical frame at p_i:
            nc: patch center direction
            xn: local right axis
            yn: local up axis

        No GCTT gauge rotation and no PB attention bias are applied here.
        """
        del v_fov_rad  # square tangent patches use h_fov for focal length

        device = self.aug_device
        pano_tensor = pano_tensor.to(device)
        N = len(angles_deg)
        P = self.patch_size

        angles_t = torch.tensor(angles_deg, dtype=torch.float32, device=device)
        theta = torch.deg2rad(angles_t[:, 0])
        phi = torch.deg2rad(angles_t[:, 1])

        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        nc = torch.stack(
            [
                cos_phi * cos_theta,
                -cos_phi * sin_theta,
                sin_phi,
            ],
            dim=1,
        )
        xn = torch.stack(
            [
                -sin_theta,
                -cos_theta,
                torch.zeros_like(theta),
            ],
            dim=1,
        )
        yn = torch.stack(
            [
                -sin_phi * cos_theta,
                sin_phi * sin_theta,
                cos_phi,
            ],
            dim=1,
        )

        L = (P / 2.0) / torch.tan(
            torch.tensor(h_fov_rad / 2.0, device=device, dtype=torch.float32)
        )

        uu = self._pixel_u.to(device)
        vv = self._pixel_v.to(device)

        pts = (
            uu.view(1, -1, 1) * xn.unsqueeze(1)
            + vv.view(1, -1, 1) * yn.unsqueeze(1)
            + L * nc.unsqueeze(1)
        )

        px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
        p_norm = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-8)

        theta_odi = torch.atan2(-py, px)
        phi_odi = torch.asin(torch.clamp(pz / p_norm, -1.0, 1.0))

        grid_x = theta_odi / torch.pi
        grid_y = -phi_odi / (torch.pi / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, P, P, 2)

        pano_batch = pano_tensor.unsqueeze(0).expand(N, -1, -1, -1)
        patches = F.grid_sample(
            pano_batch,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        return patches

    def _load_panorama(self, idx: int) -> Tuple[torch.Tensor, float]:
        path, _ = self.image_folder.samples[idx]
        with open(path, "rb") as f:
            pano_pil = Image.open(f).convert("RGB")

        if pano_pil.size != (self.pano_w, self.pano_h):
            pano_pil = pano_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)

        if self.color_jitter is not None and np.random.rand() < 0.8:
            pano_pil = self.color_jitter(pano_pil)

        if self.blur is not None and np.random.rand() < 0.1:
            pano_pil = self.blur(pano_pil)

        roll_shift_deg = 0.0
        if self.use_horizontal_roll:
            shift = np.random.randint(0, self.pano_w)
            pano_pil = Image.fromarray(np.roll(np.asarray(pano_pil), shift, axis=1))
            roll_shift_deg = (shift / self.pano_w) * 360.0

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

        return pano_tensor, roll_shift_deg

    @staticmethod
    def _random_erasing_batch(patches: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0.0:
            return patches
        # Apply erasing independently per tangent view, after normalization.
        eraser = transforms.RandomErasing(p=p, value="random")
        return torch.stack([eraser(patch) for patch in patches], dim=0)

    def __len__(self) -> int:
        return len(self.image_folder.samples)

    def __getitem__(self, idx: int):
        path, label = self.image_folder.samples[idx]
        N = self.u_num * self.v_num

        try:
            pano_tensor, roll_shift_deg = self._load_panorama(idx)

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

            patches = self._extract_patches_no_gctt(
                pano_tensor,
                angles_list,
                curr_h_fov_rad,
                curr_v_fov_rad,
            )

            mean = self._norm_mean.to(patches.device)
            std = self._norm_std.to(patches.device)
            patches = (patches - mean) / std
            patches = self._random_erasing_batch(patches, self.reprob)

            angles = torch.tensor(angles_list, dtype=torch.float32, device=patches.device)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            patches = torch.zeros(N, 3, self.patch_size, self.patch_size)
            angles = torch.zeros(N, 2)

        label = torch.tensor(label, dtype=torch.long, device=patches.device)

        # Return CPU tensors by default; the engine moves them to the training device.
        if self.return_device.type == "cpu":
            return (
                patches.detach().cpu(),
                angles.detach().cpu(),
                label.detach().cpu(),
            )

        return (
            patches.to(self.return_device, non_blocking=True),
            angles.to(self.return_device, non_blocking=True),
            label.to(self.return_device, non_blocking=True),
        )


def build_classification_dataset(is_train, args, class_to_idx=None):
    dynamic_img_size = args.pano_h // args.grid_height

    dataset = PanoClassificationDataset(
        root_dir=args.data_path if is_train else args.val_data_path,
        grid_height=args.grid_height,
        img_size=dynamic_img_size,
        pano_h=args.pano_h,
        pano_w=args.pano_w,
        is_train=is_train,
        class_to_idx=class_to_idx,
        use_color_jitter=getattr(args, "use_color_jitter", True),
        use_blur=getattr(args, "use_blur", True),
        use_full_pose3d=getattr(args, "use_full_pose3d", True),
        use_horizontal_roll=getattr(args, "use_horizontal_roll", False),
        multiscale_sampling=getattr(args, "multiscale_sampling", True),
        angle_jitter_deg=getattr(args, "angle_jitter_deg", 5.0),
        aug_device=getattr(args, "aug_device", "cpu"),
        return_device="cpu",
        reprob=getattr(args, "reprob", 0.0),
    )
    return dataset
