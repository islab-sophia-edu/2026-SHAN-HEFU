import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import util.odi_processing as odi  # kept for compatibility with your existing project


class ERPRotator:
    def __init__(self, h, w, device='cpu'):
        self.h, self.w = h, w
        self.device = torch.device(device)

        v, u = torch.meshgrid(
            torch.linspace(1, -1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        self.theta = u * np.pi
        self.phi = v * np.pi / 2
        self.xyz = torch.stack([
            torch.cos(self.phi) * torch.cos(self.theta),
            -torch.cos(self.phi) * torch.sin(self.theta),
            torch.sin(self.phi)
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
                           [0.,          0.,         1.]], device=self.device, dtype=torch.float32)
        Ry = torch.tensor([[ np.cos(p_), 0., np.sin(p_)],
                           [ 0.,         1., 0.        ],
                           [-np.sin(p_), 0., np.cos(p_)]], device=self.device, dtype=torch.float32)
        Rx = torch.tensor([[1., 0.,          0.         ],
                           [0., np.cos(r_), -np.sin(r_) ],
                           [0., np.sin(r_),  np.cos(r_) ]], device=self.device, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        xyz_rot = torch.matmul(self.xyz, R.T)
        grid = torch.stack([
            torch.atan2(xyz_rot[:, 1], xyz_rot[:, 0]) / np.pi,
            -(torch.asin(torch.clamp(xyz_rot[:, 2], -1.0, 1.0)) / (np.pi / 2.0))
        ], dim=-1).view(1, self.h, self.w, 2)

        rotated = F.grid_sample(img_4d, grid, mode='bicubic',
                                padding_mode='border', align_corners=True)
        return rotated.squeeze(0).to(orig_device)

    def rotate_image(self, image_np, yaw=0.0, pitch=0.0, roll=0.0):
        t = torch.from_numpy(image_np).float().div(255.0).permute(2, 0, 1)
        r = self.rotate_tensor(t, yaw, pitch, roll)
        return (r.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


class PanoramicDataset(Dataset):
    """
    Returns:
        views, angles, gauge_angles, weights

    Oriented-SPE / GCTT data contract:
        gauge_angles stores the in-plane tangent-frame gauge psi_i in degrees.
        During training, the local tangent frame can be rotated by a global
        per-image gauge and optional per-token local gauge.

        The model should NOT encode psi_i with a standalone Fourier GE(psi_i).
        Instead, psi_i is passed together with (theta_i, phi_i) so the positional
        encoder can construct an oriented spherical frame SPE(theta_i, phi_i, psi_i).
    """

    def __init__(self, root_dir,
                 grid_height=4, pano_h=512, pano_w=1024,
                 multiscale_sampling=True, use_full_pose3d=True,
                 use_color_jitter=True, use_blur=True,
                 use_horizontal_roll=False,
                 is_training=True, img_size=None,
                 angle_jitter_deg=5.0,
                 aug_device='cuda',
                 # --- GCTT options ---
                 use_gctt=True,
                 gctt_gauge_jitter_deg=30.0,
                 gctt_local_gauge_jitter_deg=0.0):

        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.is_training = is_training

        self.pano_h, self.pano_w = pano_h, pano_w
        self.grid_height = grid_height
        self.patch_size = img_size if img_size is not None else pano_h // grid_height

        self.v_num = grid_height
        self.u_num = 2 * grid_height
        self.base_v_fov_deg = 180.0 / self.v_num
        self.base_h_fov_deg = 360.0 / self.u_num
        self.base_angles_rad = self._generate_grid_angles_rad()

        self.multiscale_sampling = multiscale_sampling and is_training
        self.use_full_pose3d = use_full_pose3d and is_training
        self.use_horizontal_roll = use_horizontal_roll and is_training
        self.angle_jitter_deg = angle_jitter_deg if is_training else 0.0

        # GCTT: use a global per-image gauge by default. This keeps all tangent
        # frames in one image mutually consistent. Optional local jitter can be
        # enabled for stronger augmentation.
        self.use_gctt = use_gctt
        self.gctt_gauge_jitter_deg = float(gctt_gauge_jitter_deg) if is_training else 0.0
        self.gctt_local_gauge_jitter_deg = float(gctt_local_gauge_jitter_deg) if is_training else 0.0

        self.color_jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if (use_color_jitter and is_training) else None
        )
        self.blur = (
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))
            if (use_blur and is_training) else None
        )

        self.aug_device = torch.device(aug_device if torch.cuda.is_available() else 'cpu')

        P = self.patch_size
        u_lin = torch.linspace(-(P - 1) / 2.0, (P - 1) / 2.0, P)
        v_lin = torch.linspace((P - 1) / 2.0, -(P - 1) / 2.0, P)
        uu, vv = torch.meshgrid(u_lin, v_lin, indexing='xy')
        self._pixel_u = uu.reshape(-1)
        self._pixel_v = vv.reshape(-1)

        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self._rotator, self._rotator_h, self._rotator_w = None, None, None

    def _generate_grid_angles_rad(self):
        phis = np.linspace(np.pi / 2 - np.radians(self.base_v_fov_deg) / 2,
                           -np.pi / 2 + np.radians(self.base_v_fov_deg) / 2,
                           self.v_num)
        thetas = np.linspace(-np.pi, np.pi, self.u_num, endpoint=False)
        grid_phis, grid_thetas = np.meshgrid(phis, thetas, indexing='ij')
        return np.stack([grid_thetas.flatten(), grid_phis.flatten()], axis=1)

    def _get_rotator(self, h, w):
        if self._rotator is None or self._rotator_h != h or self._rotator_w != w:
            self._rotator = ERPRotator(h, w, device=str(self.aug_device))
            self._rotator_h, self._rotator_w = h, w
        return self._rotator

    def _extract_patches_gpu(self,
                             pano_tensor: torch.Tensor,
                             angles_deg: list,
                             h_fov_rad: float,
                             v_fov_rad: float,
                             gauge_angles_deg: list = None) -> torch.Tensor:
        """
        Extract tangent patches with an explicit local gauge angle.

        The canonical tangent frame is:
            nc: patch center direction
            xn: local right direction
            yn: local up direction

        GCTT rotates the tangent frame inside T_p S^2:
            x_g = cos(psi) * xn + sin(psi) * yn
            y_g = -sin(psi) * xn + cos(psi) * yn

        psi is returned to the model as gauge_angles. The model then constructs
        SPE(theta_i, phi_i, psi_i), so patch orientation is represented as part
        of the same oriented spherical state, not as an independent GE(psi_i).
        """
        device = self.aug_device
        N = len(angles_deg)
        P = self.patch_size

        angles_t = torch.tensor(angles_deg, dtype=torch.float32, device=device)
        theta = torch.deg2rad(angles_t[:, 0])
        phi = torch.deg2rad(angles_t[:, 1])

        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        nc = torch.stack([cos_phi * cos_theta,
                          -cos_phi * sin_theta,
                          sin_phi], dim=1)
        xn = torch.stack([-sin_theta,
                          -cos_theta,
                          torch.zeros_like(theta)], dim=1)
        yn = torch.stack([-sin_phi * cos_theta,
                          sin_phi * sin_theta,
                          cos_phi], dim=1)

        # --- GCTT frame rotation ---
        if gauge_angles_deg is None:
            gauge = torch.zeros(N, dtype=torch.float32, device=device)
        else:
            gauge = torch.tensor(gauge_angles_deg, dtype=torch.float32, device=device).view(N)
        psi = torch.deg2rad(gauge)
        c, s = torch.cos(psi).unsqueeze(1), torch.sin(psi).unsqueeze(1)
        xg = c * xn + s * yn
        yg = -s * xn + c * yn
        # ---------------------------

        # Original implementation used horizontal FoV to determine focal length.
        # Kept for compatibility with your square tangent patches.
        L = (P / 2.0) / torch.tan(
            torch.tensor(h_fov_rad / 2.0, device=device, dtype=torch.float32))

        uu = self._pixel_u.to(device)
        vv = self._pixel_v.to(device)

        pts = (uu.view(1, -1, 1) * xg.unsqueeze(1) +
               vv.view(1, -1, 1) * yg.unsqueeze(1) +
               L * nc.unsqueeze(1))

        px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]

        p_norm = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-8)
        theta_odi = torch.atan2(-py, px)
        phi_odi = torch.asin(torch.clamp(pz / p_norm, -1.0, 1.0))

        grid_x = theta_odi / torch.pi
        grid_y = -phi_odi / (torch.pi / 2.0)

        grid = torch.stack([grid_x, grid_y], dim=-1).view(N, P, P, 2)

        pano_batch = pano_tensor.unsqueeze(0).expand(N, -1, -1, -1)
        patches = F.grid_sample(pano_batch, grid,
                                mode='bicubic',
                                padding_mode='border',
                                align_corners=True)
        return patches

    def _load_and_process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        pano_pil = Image.open(img_path).convert('RGB')

        if pano_pil.size != (self.pano_w, self.pano_h):
            pano_pil = pano_pil.resize((self.pano_w, self.pano_h), Image.BICUBIC)

        if self.color_jitter and np.random.rand() < 0.8:
            pano_pil = self.color_jitter(pano_pil)
        if self.blur and np.random.rand() < 0.1:
            pano_pil = self.blur(pano_pil)

        roll_shift_deg = 0.0
        if self.use_horizontal_roll:
            shift = np.random.randint(0, self.pano_w)
            pano_pil = Image.fromarray(
                np.roll(np.array(pano_pil), shift, axis=1)
            )
            roll_shift_deg = (shift / self.pano_w) * 360.0

        pano_tensor = transforms.ToTensor()(pano_pil).to(self.aug_device)

        if self.use_full_pose3d:
            yaw_shift = np.random.uniform(0.0, 360.0)
            pitch_shift = np.random.uniform(-30.0, 30.0)
            roll_shift = np.random.uniform(-30.0, 30.0)
            rotator = self._get_rotator(self.pano_h, self.pano_w)
            pano_tensor = rotator.rotate_tensor(
                pano_tensor, yaw_shift, pitch_shift, roll_shift
            )

        return pano_tensor, roll_shift_deg

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            pano_tensor, roll_shift_deg = self._load_and_process_image(idx)

            scale = np.random.uniform(0.8, 1.2) if self.multiscale_sampling else 1.0
            curr_h_fov_rad = np.radians(self.base_h_fov_deg * scale)
            curr_v_fov_rad = np.radians(self.base_v_fov_deg * scale)

            # GCTT: global gauge is shared by all patches in an image, preserving
            # a coherent tangent-frame convention. Local gauge jitter is optional.
            if self.use_gctt and self.is_training and self.gctt_gauge_jitter_deg > 0:
                global_gauge_deg = np.random.uniform(-self.gctt_gauge_jitter_deg,
                                                     self.gctt_gauge_jitter_deg)
            else:
                global_gauge_deg = 0.0

            angles_list = []
            gauge_list = []
            for base_theta, base_phi in self.base_angles_rad:
                jitter_t = np.radians(
                    np.random.uniform(-self.angle_jitter_deg, self.angle_jitter_deg))
                jitter_p = np.radians(
                    np.random.uniform(-self.angle_jitter_deg, self.angle_jitter_deg))

                roll_rad = np.radians(roll_shift_deg)
                curr_theta = base_theta + jitter_t + roll_rad
                curr_phi = np.clip(base_phi + jitter_p,
                                   -np.pi / 2 + curr_v_fov_rad / 2,
                                   np.pi / 2 - curr_v_fov_rad / 2)
                angles_list.append([np.degrees(curr_theta), np.degrees(curr_phi)])

                if self.use_gctt and self.is_training and self.gctt_local_gauge_jitter_deg > 0:
                    local_gauge_deg = np.random.uniform(-self.gctt_local_gauge_jitter_deg,
                                                        self.gctt_local_gauge_jitter_deg)
                else:
                    local_gauge_deg = 0.0
                gauge_list.append(global_gauge_deg + local_gauge_deg)

            patches = self._extract_patches_gpu(
                pano_tensor,
                angles_list,
                curr_h_fov_rad,
                curr_v_fov_rad,
                gauge_angles_deg=gauge_list if self.use_gctt else None,
            )

            mean = self._norm_mean.to(self.aug_device)
            std = self._norm_std.to(self.aug_device)
            patches = (patches - mean) / std

            views = patches.cpu()
            angles = torch.tensor(angles_list, dtype=torch.float32)
            gauge_angles = torch.tensor(gauge_list, dtype=torch.float32).unsqueeze(-1)

            weights = torch.cos(torch.deg2rad(angles[:, 1]))
            weights = weights / (weights.mean() + 1e-6)

            return views, angles, gauge_angles, weights

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            import traceback
            traceback.print_exc()
            dummy_n = self.u_num * self.v_num
            return (torch.zeros(dummy_n, 3, self.patch_size, self.patch_size),
                    torch.zeros(dummy_n, 2),
                    torch.zeros(dummy_n, 1),
                    torch.zeros(dummy_n))
