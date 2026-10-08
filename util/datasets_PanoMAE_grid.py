import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ERPRotator:
    """SO(3) rotation on a full ERP image.

    This is kept as an optional full-panorama augmentation. It is not used for
    tokenization. The grid tokens are always plain rectangular ERP patches.
    """

    def __init__(self, h, w, device='cpu'):
        self.h, self.w = h, w
        self.device = torch.device(device)

        v, u = torch.meshgrid(
            torch.linspace(1, -1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij',
        )
        self.theta = u * np.pi
        self.phi = v * np.pi / 2
        self.xyz = torch.stack([
            torch.cos(self.phi) * torch.cos(self.theta),
            -torch.cos(self.phi) * torch.sin(self.theta),
            torch.sin(self.phi),
        ], dim=-1).view(-1, 3)

    def rotate_tensor(self, img_tensor: torch.Tensor,
                      yaw=0.0, pitch=0.0, roll=0.0) -> torch.Tensor:
        if yaw == 0 and pitch == 0 and roll == 0:
            return img_tensor

        orig_device = img_tensor.device
        img_4d = img_tensor.to(self.device).unsqueeze(0)

        y_, p_, r_ = np.radians(yaw), np.radians(pitch), np.radians(roll)
        Rz = torch.tensor([[np.cos(y_), -np.sin(y_), 0.],
                           [np.sin(y_),  np.cos(y_), 0.],
                           [0.,          0.,         1.]],
                          device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[ np.cos(p_), 0., np.sin(p_)],
                           [ 0.,         1., 0.        ],
                           [-np.sin(p_), 0., np.cos(p_)]],
                          device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1., 0.,          0.         ],
                           [0., np.cos(r_), -np.sin(r_) ],
                           [0., np.sin(r_),  np.cos(r_) ]],
                          device=self.device, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([
            torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi,
            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1.0, 1.0)) / (np.pi / 2.0)),
        ], dim=-1).view(1, self.h, self.w, 2)

        rotated = F.grid_sample(
            img_4d,
            grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True,
        )
        return rotated.squeeze(0).to(orig_device)


class PanoramicDataset(Dataset):
    """Pure ERP-grid pre-training dataset for MAE-style ablation.

    Tokenization:
        - No ODI.
        - No tangent-plane projection.
        - The full ERP image is split into a fixed row-major grid, exactly like
          MAE patchify/unfold.

    Returns:
        PB / non-GCTT mode:
            views, angles, weights
        GCTT-compatible mode:
            views, angles, gauge_angles, weights

    Notes:
        - `angles` are patch-center ERP spherical coordinates in degrees:
          [longitude/theta, latitude/phi].
        - `gauge_angles` are fixed to zero for clean grid tokenization. Random
          gauge jitter would change the positional state without rotating the
          rectangular ERP patch content, so it is deliberately not applied here.
    """

    def __init__(self, root_dir,
                 grid_height=16, pano_h=512, pano_w=1024,
                 multiscale_sampling=False, use_full_pose3d=False,
                 use_color_jitter=True, use_blur=True,
                 use_horizontal_roll=False,
                 is_training=True, img_size=None,
                 angle_jitter_deg=0.0,
                 aug_device='cuda',
                 # --- compatibility flags ---
                 use_gctt=True,
                 gctt_gauge_jitter_deg=0.0,
                 gctt_local_gauge_jitter_deg=0.0):
        del multiscale_sampling, angle_jitter_deg, gctt_gauge_jitter_deg, gctt_local_gauge_jitter_deg

        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.is_training = is_training

        self.pano_h = int(pano_h)
        self.pano_w = int(pano_w)
        self.grid_height = int(grid_height)
        self.v_num = self.grid_height
        self.u_num = 2 * self.grid_height

        if self.pano_h % self.v_num != 0:
            raise ValueError(
                f'pano_h={self.pano_h} must be divisible by grid_height={self.v_num}.'
            )
        if self.pano_w % self.u_num != 0:
            raise ValueError(
                f'pano_w={self.pano_w} must be divisible by 2*grid_height={self.u_num}.'
            )

        patch_h = self.pano_h // self.v_num
        patch_w = self.pano_w // self.u_num
        if img_size is not None and (patch_h != int(img_size) or patch_w != int(img_size)):
            raise ValueError(
                f'img_size={img_size} is inconsistent with ERP grid patch size '
                f'{patch_h}x{patch_w}. Use img_size={patch_h} or adjust pano/grid.'
            )
        if patch_h != patch_w:
            raise ValueError(
                f'This MAE head predicts square patches, but grid patch is {patch_h}x{patch_w}. '
                f'Use a 2:1 panorama with u_num=2*grid_height.'
            )
        self.patch_size = patch_h

        self.use_full_pose3d = bool(use_full_pose3d and is_training)
        self.use_horizontal_roll = bool(use_horizontal_roll and is_training)
        self.use_gctt = bool(use_gctt)

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if (use_color_jitter and is_training) else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if (use_blur and is_training) else None
        )

        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else 'cpu')

        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.base_angles_deg = torch.tensor(self._generate_grid_angles_deg(), dtype=torch.float32)
        self.base_weights = torch.cos(torch.deg2rad(self.base_angles_deg[:, 1]))
        self.base_weights = self.base_weights / (self.base_weights.mean() + 1e-6)

        self._rotator = None
        self._rotator_h = None
        self._rotator_w = None

    def _generate_grid_angles_deg(self):
        """Return row-major patch-center coordinates [theta, phi] in degrees."""
        rows = np.arange(self.v_num, dtype=np.float32)
        cols = np.arange(self.u_num, dtype=np.float32)

        # Pixel-center / patch-center locations on ERP.
        theta = -180.0 + (cols + 0.5) * (360.0 / self.u_num)
        phi = 90.0 - (rows + 0.5) * (180.0 / self.v_num)
        grid_phi, grid_theta = np.meshgrid(phi, theta, indexing='ij')
        return np.stack([grid_theta.reshape(-1), grid_phi.reshape(-1)], axis=1)

    def _get_rotator(self, h, w):
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h, self._rotator_w = h, w
        return self._rotator

    def _load_and_process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        pano_pil = Image.open(img_path).convert('RGB')

        if pano_pil.size != (self.pano_w, self.pano_h):
            pano_pil = pano_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)

        if self.color_jitter and np.random.rand() < 0.8:
            pano_pil = self.color_jitter(pano_pil)
        if self.blur and np.random.rand() < 0.1:
            pano_pil = self.blur(pano_pil)

        if self.use_horizontal_roll:
            shift = np.random.randint(0, self.pano_w)
            pano_pil = Image.fromarray(np.roll(np.array(pano_pil), shift, axis=1))

        pano_tensor = transforms.ToTensor()(pano_pil).to(self.aug_device)

        if self.use_full_pose3d:
            yaw_shift = np.random.uniform(0.0, 360.0)
            pitch_shift = np.random.uniform(-30.0, 30.0)
            roll_shift = np.random.uniform(-30.0, 30.0)
            rotator = self._get_rotator(self.pano_h, self.pano_w)
            pano_tensor = rotator.rotate_tensor(
                pano_tensor, yaw_shift, pitch_shift, roll_shift
            )

        return pano_tensor

    def _patchify_grid(self, pano_tensor: torch.Tensor) -> torch.Tensor:
        """Split [3,H,W] ERP into row-major [N,3,P,P] grid patches."""
        C, H, W = pano_tensor.shape
        P = self.patch_size
        if H != self.pano_h or W != self.pano_w:
            raise ValueError(f'Unexpected tensor size {H}x{W}; expected {self.pano_h}x{self.pano_w}.')

        patches = (
            pano_tensor
            .view(C, self.v_num, P, self.u_num, P)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
            .view(self.v_num * self.u_num, C, P, P)
        )
        return patches

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            pano_tensor = self._load_and_process_image(idx)
            patches = self._patchify_grid(pano_tensor)

            mean = self._norm_mean.to(self.aug_device)
            std = self._norm_std.to(self.aug_device)
            patches = (patches - mean) / std

            views = patches.cpu()
            angles = self.base_angles_deg.clone()
            weights = self.base_weights.clone()

            if self.use_gctt:
                gauge_angles = torch.zeros(angles.shape[0], 1, dtype=torch.float32)
                return views, angles, gauge_angles, weights
            return views, angles, weights

        except Exception as e:
            print(f'Error processing item {idx}: {e}')
            import traceback
            traceback.print_exc()
            dummy_n = self.u_num * self.v_num
            views = torch.zeros(dummy_n, 3, self.patch_size, self.patch_size)
            angles = torch.zeros(dummy_n, 2)
            weights = torch.zeros(dummy_n)
            if self.use_gctt:
                gauge_angles = torch.zeros(dummy_n, 1)
                return views, angles, gauge_angles, weights
            return views, angles, weights
