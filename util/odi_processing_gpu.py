import numpy as np
import torch
import torch.nn.functional as F
import math


def rnd(x):
    if type(x) is np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (x + 0.5).astype(int)
    else:
        return round(x)


def polar(cord):
    if cord.ndim == 1:
        P = np.linalg.norm(cord)
    else:
        P = np.linalg.norm(cord, axis=0)
    P_safe = np.maximum(P, 1e-8) if isinstance(P, np.ndarray) else max(float(P), 1e-8)
    phi = np.arcsin(np.clip(cord[2] / P_safe, -1.0, 1.0))
    r_xy = np.sqrt(cord[0] ** 2 + cord[1] ** 2)
    r_xy_safe = np.maximum(r_xy, 1e-8) if isinstance(r_xy, np.ndarray) else max(float(r_xy), 1e-8)
    cos_val = np.clip(cord[0] / r_xy_safe, -1.0, 1.0)
    theta_pos = np.arccos(cos_val)
    theta_neg = -theta_pos
    if isinstance(cord[1], np.ndarray):
        theta = np.where(cord[1] > 0, theta_neg, theta_pos)
    else:
        theta = theta_neg if cord[1] > 0 else theta_pos
    return [theta, phi]


def limit_values(x, r, xcopy=1):
    ret = x.copy() if xcopy == 1 else x
    ret[ret < r[0]] = r[0]
    ret[ret > r[1]] = r[1]
    return ret


class CameraPrm:
    def __init__(self, camera_angle, image_plane_size=None, view_angle=None, L=None):
        self.camera_angle = camera_angle
        if view_angle is None:
            self.image_plane_size = image_plane_size
            self.L = L
            self.view_angle = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L))
        elif image_plane_size is None:
            self.view_angle = view_angle
            self.L = L
            self.image_plane_size = 2.0 * L * np.tan(np.array(view_angle) / 2.0)
        else:
            self.image_plane_size = image_plane_size
            self.view_angle = view_angle
            L_arr = (np.array(image_plane_size) / 2.0) / np.tan(np.array(view_angle) / 2.0)
            self.L = L_arr[0]

        self.nc = np.array([
            np.cos(camera_angle[1]) * np.cos(camera_angle[0]),
            -np.cos(camera_angle[1]) * np.sin(camera_angle[0]),
            np.sin(camera_angle[1])
        ])
        self.c0 = self.L * self.nc
        self.xn = np.array([
            -np.sin(camera_angle[0]),
            -np.cos(camera_angle[0]),
            0.0
        ])
        self.yn = np.array([
            -np.sin(camera_angle[1]) * np.cos(camera_angle[0]),
            np.sin(camera_angle[1]) * np.sin(camera_angle[0]),
            np.cos(camera_angle[1])
        ])
        [c1, r1] = np.meshgrid(
            np.arange(0, rnd(self.image_plane_size[0])),
            np.arange(0, rnd(self.image_plane_size[1]))
        )
        img_cord = [c1 - self.image_plane_size[0] / 2.0, -r1 + self.image_plane_size[1] / 2.0]
        self.p = self.get_3Dcordinate(img_cord)
        self.polar_omni_cord = polar(self.p)

    def get_3Dcordinate(self, c):
        [xp, yp] = c
        if type(xp) is np.ndarray:
            return (xp * self.xn.reshape((3, 1, 1)) +
                    yp * self.yn.reshape((3, 1, 1)) +
                    np.ones(xp.shape) * self.c0.reshape((3, 1, 1)))
        else:
            return xp * self.xn + yp * self.yn + self.c0


def _build_batch_grids(angles_rad, h_fov_deg, v_fov_deg, patch_h, patch_w, device):
    N = len(angles_rad)
    theta_r = torch.tensor([a[0] for a in angles_rad], dtype=torch.float32, device=device)
    phi_r   = torch.tensor([a[1] for a in angles_rad], dtype=torch.float32, device=device)

    L = (patch_w / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)

    nc = torch.stack([
        torch.cos(phi_r) * torch.cos(theta_r),
        -torch.cos(phi_r) * torch.sin(theta_r),
        torch.sin(phi_r)
    ], dim=1)
    xn = torch.stack([
        -torch.sin(theta_r),
        -torch.cos(theta_r),
        torch.zeros(N, device=device)
    ], dim=1)
    yn = torch.stack([
        -torch.sin(phi_r) * torch.cos(theta_r),
        torch.sin(phi_r) * torch.sin(theta_r),
        torch.cos(phi_r)
    ], dim=1)

    r1 = torch.arange(patch_h, dtype=torch.float32, device=device)
    c1 = torch.arange(patch_w, dtype=torch.float32, device=device)
    R, C = torch.meshgrid(r1, c1, indexing='ij')
    xp = (C - patch_w / 2.0).view(1, 1, patch_h, patch_w)
    yp = (-R + patch_h / 2.0).view(1, 1, patch_h, patch_w)

    p = xp * xn.view(N, 3, 1, 1) + yp * yn.view(N, 3, 1, 1) + L * nc.view(N, 3, 1, 1)

    P_norm = torch.norm(p, dim=1).clamp(min=1e-8)
    phi_omni = torch.asin(torch.clamp(p[:, 2] / P_norm, -1.0, 1.0))
    r_xy = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2).clamp(min=1e-8)
    acos_val = torch.acos(torch.clamp(p[:, 0] / r_xy, -1.0, 1.0))
    theta_omni = torch.where(p[:, 1] > 0, -acos_val, acos_val)

    grid_x = theta_omni / math.pi
    grid_y = -2.0 * phi_omni / math.pi
    return torch.stack([grid_x, grid_y], dim=-1)


class OmniImage:
    def __init__(self, omni_image, device=None):
        if type(omni_image) is str:
            from PIL import Image as _PILImage
            omni_image = np.array(_PILImage.open(omni_image).convert('RGB'))
        self.omni_image = omni_image
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        img_f32 = (omni_image.astype(np.float32) / 255.0
                   if omni_image.dtype == np.uint8 else omni_image.astype(np.float32))
        self._tensor = torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0).to(device)

    def extract(self, camera_prm):
        theta = camera_prm.polar_omni_cord[0].astype(np.float32)
        phi   = camera_prm.polar_omni_cord[1].astype(np.float32)
        gx = torch.from_numpy(theta / math.pi)
        gy = torch.from_numpy(-2.0 * phi / math.pi)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(self._device)
        out = F.grid_sample(self._tensor, grid, mode='bilinear',
                            padding_mode='border', align_corners=False)
        return (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def batch_extract(self, camera_prms):
        grids = []
        for prm in camera_prms:
            theta = prm.polar_omni_cord[0].astype(np.float32)
            phi   = prm.polar_omni_cord[1].astype(np.float32)
            grids.append(np.stack([theta / math.pi, -2.0 * phi / math.pi], axis=-1))
        grid_t = torch.from_numpy(np.stack(grids)).to(self._device)
        N = grid_t.shape[0]
        out = F.grid_sample(self._tensor.expand(N, -1, -1, -1), grid_t,
                            mode='bilinear', padding_mode='border', align_corners=False)
        arr = (out.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return list(arr)

    def batch_extract_from_angles(self, angles_rad, h_fov_deg, v_fov_deg, patch_size):
        P = patch_size if isinstance(patch_size, int) else patch_size[0]
        W = patch_size if isinstance(patch_size, int) else patch_size[1]
        grid_t = _build_batch_grids(angles_rad, h_fov_deg, v_fov_deg, P, W, self._device)
        N = grid_t.shape[0]
        out = F.grid_sample(self._tensor.expand(N, -1, -1, -1), grid_t,
                            mode='bilinear', padding_mode='border', align_corners=False)
        arr = (out.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return list(arr)


def extract_patches_from_pano(pano_np, h_num, v_num, h_fov_deg, v_fov_deg, out_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imc = OmniImage(pano_np, device=device)

    thetas = np.linspace(-np.pi, np.pi, h_num, endpoint=False)
    v_fov_rad = np.radians(v_fov_deg)
    phis = np.linspace(
        np.pi / 2.0 - v_fov_rad / 2.0,
        -np.pi / 2.0 + v_fov_rad / 2.0,
        v_num
    )
    angles_rad = [(float(t), float(p)) for p in phis for t in thetas]
    return imc.batch_extract_from_angles(angles_rad, h_fov_deg, v_fov_deg, out_size)


def embed_patches_to_pano(patches_np_list, pano_size, h_num, v_num, h_fov_deg, v_fov_deg):
    pano_canvas = np.zeros((pano_size[0], pano_size[1], 3), dtype=np.float32)
    mask_all = np.zeros((pano_size[0], pano_size[1]), dtype=np.int32)

    thetas = np.linspace(-np.pi, np.pi, h_num, endpoint=False)
    v_fov_rad = np.radians(v_fov_deg)
    phis = np.linspace(np.pi / 2.0 - v_fov_rad / 2.0, -np.pi / 2.0 + v_fov_rad / 2.0, v_num)

    c_omni, r_omni = np.meshgrid(np.arange(pano_size[1]), np.arange(pano_size[0]))
    theta_omni = (2.0 * c_omni / float(pano_size[1] - 1) - 1.0) * np.pi
    phi_omni = (0.5 - r_omni / float(pano_size[0] - 1)) * np.pi

    pn = np.stack([
        np.cos(phi_omni) * np.cos(theta_omni),
        -np.cos(phi_omni) * np.sin(theta_omni),
        np.sin(phi_omni)
    ], axis=2)

    patch_idx = 0
    for phi_c in phis:
        for theta_c in thetas:
            patch = patches_np_list[patch_idx]
            if patch.dtype == np.uint8:
                patch = patch.astype(np.float32) / 255.0
            elif patch.dtype != np.float32:
                patch = patch.astype(np.float32)

            h_patch, w_patch = patch.shape[:2]
            prm = CameraPrm(
                camera_angle=[theta_c, phi_c],
                view_angle=[np.radians(h_fov_deg), np.radians(v_fov_deg)],
                image_plane_size=[w_patch, h_patch]
            )
            L, nc, xn, yn = prm.L, prm.nc, prm.xn, prm.yn

            cos_alpha = np.dot(pn, nc)
            mask = cos_alpha >= 2 * L / np.sqrt(w_patch ** 2 + h_patch ** 2 + 4 * L ** 2)

            xp = np.zeros((pano_size[0], pano_size[1]))
            yp = np.zeros((pano_size[0], pano_size[1]))

            if np.any(mask):
                pn_m = pn[mask]
                r_m = L / np.dot(pn_m, nc)
                xp[mask] = r_m * np.dot(pn_m, xn)
                yp[mask] = r_m * np.dot(pn_m, yn)

            mask = mask & (xp > -w_patch / 2.0) & (xp < w_patch / 2.0) & \
                          (yp > -h_patch / 2.0) & (yp < h_patch / 2.0)

            if not np.any(mask):
                patch_idx += 1
                continue

            c1_int = limit_values(rnd(w_patch / 2.0 + xp - 0.5), (0, w_patch - 1), 0)
            r1_int = limit_values(rnd(h_patch / 2.0 - yp - 0.5), (0, h_patch - 1), 0)
            pano_canvas[mask] = patch[r1_int, c1_int][mask]
            mask_all[mask] += 1
            patch_idx += 1

    overlapping = np.sum(mask_all > 1)
    missing = np.sum(mask_all == 0)
    total = pano_size[0] * pano_size[1]
    print(f'Overlapping dots: {overlapping}/{total}, Missing dots: {missing}/{total}')
    return pano_canvas


def make_grid_gpu(tensor_list, grid_width, padding=4, bg_color=0):
    if isinstance(tensor_list, list):
        if len(tensor_list) == 0:
            return None
        patches = torch.stack(tensor_list)
    else:
        patches = tensor_list

    N, C, H, W = patches.shape
    grid_height = (N + grid_width - 1) // grid_width
    canvas_h = grid_height * (H + padding) + padding
    canvas_w = grid_width * (W + padding) + padding
    canvas = torch.full((C, canvas_h, canvas_w), bg_color,
                        dtype=patches.dtype, device=patches.device)
    for i in range(N):
        row, col = i // grid_width, i % grid_width
        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        canvas[:, y:y + H, x:x + W] = patches[i]
    return canvas


def embed_patches_to_pano_gpu(patches_tensor, pano_h, pano_w,
                               h_num, v_num, h_fov_deg, v_fov_deg, angles):
    device = patches_tensor.device
    N, C, H_patch, W_patch = patches_tensor.shape

    pano_canvas = torch.zeros((C, pano_h, pano_w), device=device, dtype=torch.float32)

    y_range = torch.arange(pano_h, device=device, dtype=torch.float32)
    x_range = torch.arange(pano_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    theta_omni = (2.0 * grid_x / (pano_w - 1) - 1.0) * torch.pi
    phi_omni   = (0.5 - grid_y / (pano_h - 1)) * torch.pi

    cos_phi = torch.cos(phi_omni)
    pn = torch.stack([
        cos_phi * torch.cos(theta_omni),
        -cos_phi * torch.sin(theta_omni),
        torch.sin(phi_omni)
    ], dim=0)

    L = (W_patch / 2.0) / torch.tan(torch.deg2rad(
        torch.tensor(h_fov_deg / 2.0, device=device)))

    for i in range(N):
        theta_rad = torch.deg2rad(angles[i, 0])
        phi_rad   = torch.deg2rad(angles[i, 1])

        nc = torch.stack([
            torch.cos(phi_rad) * torch.cos(theta_rad),
            -torch.cos(phi_rad) * torch.sin(theta_rad),
            torch.sin(phi_rad)
        ]).view(3, 1, 1)
        xn = torch.stack([
            -torch.sin(theta_rad),
            -torch.cos(theta_rad),
            torch.tensor(0.0, device=device)
        ]).view(3, 1, 1)
        yn = torch.stack([
            -torch.sin(phi_rad) * torch.cos(theta_rad),
            torch.sin(phi_rad) * torch.sin(theta_rad),
            torch.cos(phi_rad)
        ]).view(3, 1, 1)

        cos_alpha = torch.sum(pn * nc, dim=0)
        threshold = 2 * L / torch.sqrt(torch.tensor(
            W_patch ** 2 + H_patch ** 2 + 4 * L ** 2, device=device, dtype=torch.float32))
        mask_fov = cos_alpha >= threshold
        if not mask_fov.any():
            continue

        r_dist = L / (cos_alpha + 1e-8)
        xp = r_dist * torch.sum(pn * xn, dim=0)
        yp = r_dist * torch.sum(pn * yn, dim=0)

        final_mask = mask_fov & (xp > -W_patch / 2.0) & (xp < W_patch / 2.0) & \
                                (yp > -H_patch / 2.0) & (yp < H_patch / 2.0)
        if not final_mask.any():
            continue

        c1 = (W_patch / 2.0 + xp[final_mask] - 0.5).round().long().clamp(0, W_patch - 1)
        r1 = (H_patch / 2.0 - yp[final_mask] - 0.5).round().long().clamp(0, H_patch - 1)
        pano_canvas[:, final_mask] = patches_tensor[i, :, r1, c1]

    return pano_canvas