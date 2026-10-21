"""Explicit per-patch camera orientation for a tangent/ERP normal decoder.

Tangent frames exactly follow normal_dataset_core._frames_from_angles:
  forward = (cos(phi)cos(theta), -cos(phi)sin(theta), sin(phi))
  right   = (-sin(theta), -cos(theta), 0)
  up      = (-sin(phi)cos(theta), sin(phi)sin(theta), cos(phi))
  x_g = cos(psi)right + sin(psi)up
  y_g = -sin(psi)right + cos(psi)up
The proper camera-to-ERP rotation is [x_g, -y_g, forward] with
camera coordinates (right, down, forward). A panorama patch has rotation
but no independent camera translation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderCameraPose(nn.Module):
    """Pose-conditioned FiLM and rays at four dense-decoder feature scales."""

    CHANNELS = {"neck": 512, "skip1": 256, "skip2": 128, "skip3": 64}

    def __init__(self, grid_height=16, is_grid=False, strength=0.2):
        super().__init__()
        self.grid_h = int(grid_height)
        self.grid_w = 2 * self.grid_h
        self.is_grid = bool(is_grid)
        self.strength = float(strength)
        if self.grid_h < 1 or not 0.0 < self.strength <= 1.0:
            raise ValueError("grid_height must be positive and strength must be in (0,1]")
        self.h_fov = math.radians(360.0 / self.grid_w)
        self.v_fov = math.radians(180.0 / self.grid_h)
        self.film = nn.ModuleDict({
            key: nn.Linear(9, channels * 2)
            for key, channels in self.CHANNELS.items()
        })
        self.ray_proj = nn.ModuleDict({
            key: nn.Conv2d(3, channels, kernel_size=1, bias=False)
            for key, channels in self.CHANNELS.items()
        })
        for layer in self.film.values():
            nn.init.normal_(layer.weight, std=0.035)
            nn.init.zeros_(layer.bias)
        for layer in self.ray_proj.values():
            nn.init.normal_(layer.weight, std=0.015)

    @staticmethod
    def _frames(angles, gauge=None):
        if angles.ndim != 3 or angles.shape[-1] != 2:
            raise ValueError("angles must have shape [B,N,2] (longitude,latitude in degrees)")
        theta = torch.deg2rad(angles[..., 0].float())
        phi = torch.deg2rad(angles[..., 1].float())
        ct, st = theta.cos(), theta.sin()
        cp, sp = phi.cos(), phi.sin()
        forward = torch.stack((cp * ct, -cp * st, sp), dim=-1)
        right0 = torch.stack((-st, -ct, torch.zeros_like(theta)), dim=-1)
        up0 = torch.stack((-sp * ct, sp * st, cp), dim=-1)
        if gauge is None:
            return forward, right0, up0
        psi = torch.deg2rad(gauge.float().reshape_as(theta))
        c, s = psi.cos().unsqueeze(-1), psi.sin().unsqueeze(-1)
        right = c * right0 + s * up0
        up = -s * right0 + c * up0
        return forward, right, up

    @classmethod
    def camera_rotation(cls, angles, gauge=None):
        """Return camera(right,down,forward) -> ERP, [B,N,3,3], det +1."""
        forward, right, up = cls._frames(angles, gauge)
        return torch.stack((right, -up, forward), dim=-1)

    def _check(self, angles, stage, size=None):
        if stage not in self.CHANNELS:
            raise KeyError(f"Unknown pose stage: {stage}")
        if angles.shape[1] != self.grid_h * self.grid_w:
            raise ValueError(f"Expected {self.grid_h*self.grid_w} patch angles, got {angles.shape[1]}")
        if size is not None:
            h, w = size
            if h % self.grid_h or w % self.grid_w or h // self.grid_h != w // self.grid_w:
                raise ValueError(f"Decoder feature {size} is not an aligned square-subpatch mosaic")
            return h // self.grid_h

    @staticmethod
    def _mosaic(v, grid_h, grid_w, subpixels):
        b, n, h, w, c = v.shape
        if (n, h, w) != (grid_h * grid_w, subpixels, subpixels):
            raise ValueError(f"Unexpected patch shape: {tuple(v.shape)}")
        return (v.reshape(b, grid_h, grid_w, h, w, c)
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(b, c, grid_h * h, grid_w * w))

    def _canonical_grid_angles(self, batch, device):
        """Actual center of each rectangular ERP grid cell (not tangent centers)."""
        row = torch.arange(self.grid_h, device=device, dtype=torch.float32)
        col = torch.arange(self.grid_w, device=device, dtype=torch.float32)
        lat = (0.5 - (row + 0.5) / self.grid_h) * 180.0
        lon = ((col + 0.5) / self.grid_w * 2.0 - 1.0) * 180.0
        phi, theta = torch.meshgrid(lat, lon, indexing="ij")
        return torch.stack((theta, phi), dim=-1).reshape(1, -1, 2).expand(batch, -1, -1)

    def patch_rays(self, angles, gauge, subpixels):
        """Return rays [B,3,H,W] in the same geometry as the dataset extraction."""
        s = int(subpixels)
        if s < 1:
            raise ValueError("subpixels must be positive")
        b, n, _ = angles.shape
        self._check(angles, "neck")
        device = angles.device
        if self.is_grid:
            h, w = self.grid_h * s, self.grid_w * s
            rows = torch.arange(h, device=device, dtype=torch.float32)
            cols = torch.arange(w, device=device, dtype=torch.float32)
            lat = (0.5 - (rows + 0.5) / h) * math.pi
            lon = (2.0 * (cols + 0.5) / w - 1.0) * math.pi
            cp, sp = lat.cos()[:, None], lat.sin()[:, None]
            ct, st = lon.cos()[None, :], lon.sin()[None, :]
            return torch.stack((cp * ct, -cp * st, sp.expand(h, w)), dim=0)[None].expand(b, -1, -1, -1)

        forward, right, up = self._frames(angles, gauge)
        # Same pixel-centre convention and fx/fy as _extract_patches_gpu.
        u = (2.0 * (torch.arange(s, device=device, dtype=torch.float32) + 0.5) / s - 1.0) * math.tan(self.h_fov / 2.0)
        v = (1.0 - 2.0 * (torch.arange(s, device=device, dtype=torch.float32) + 0.5) / s) * math.tan(self.v_fov / 2.0)
        rays = (forward[:, :, None, None, :]
                + u[None, None, None, :, None] * right[:, :, None, None, :]
                + v[None, None, :, None, None] * up[:, :, None, None, :])
        rays = F.normalize(rays, p=2, dim=-1)
        return self._mosaic(rays, self.grid_h, self.grid_w, s)

    @classmethod
    def local_normal_to_erp(cls, patch_normals, angles, gauge=None):
        """Rotate [B,N,3,H,W] camera-frame predictions to ERP-frame normals."""
        if patch_normals.ndim != 5 or patch_normals.shape[2] != 3:
            raise ValueError(f"Expected local normals [B,N,3,H,W], got {tuple(patch_normals.shape)}")
        if tuple(patch_normals.shape[:2]) != tuple(angles.shape[:2]):
            raise ValueError("Patch normal count and camera pose count differ")
        rot = cls.camera_rotation(angles.detach(), gauge.detach() if gauge is not None else None)
        result = torch.einsum("bnij,bnjhw->bnihw", rot.to(dtype=patch_normals.dtype), patch_normals)
        return F.normalize(result, p=2, dim=2, eps=1e-6)

    def forward(self, x, angles, gauge, stage):
        """Modulate decoder features using each patch's explicit camera rotation."""
        s = self._check(angles, stage, x.shape[-2:])
        if x.ndim != 4 or x.shape[1] != self.CHANNELS[stage]:
            raise ValueError(f"Invalid decoder feature at {stage}: {tuple(x.shape)}")
        a = angles.detach()
        if self.is_grid:
            a = self._canonical_grid_angles(angles.shape[0], angles.device)
        g = gauge.detach() if gauge is not None and not self.is_grid else None
        r = self.camera_rotation(a, g).flatten(start_dim=-2)  # [B,N,9]
        film = self.film[stage](r.to(dtype=self.film[stage].weight.dtype))
        b, n, channels_twice = film.shape
        film = (film.reshape(b, self.grid_h, self.grid_w, channels_twice)
                .permute(0, 3, 1, 2)
                .repeat_interleave(s, dim=2).repeat_interleave(s, dim=3))
        gain, bias = film.chunk(2, dim=1)
        rays = self.patch_rays(a, g, s)
        ray_term = self.ray_proj[stage](rays)
        # Unlike the old zero-initialised 1x1 branch, this uses rotation
        # and ray geometry immediately and stays learnable during fine-tuning.
        gain = gain.to(dtype=x.dtype)
        bias = bias.to(dtype=x.dtype)
        ray_term = ray_term.to(dtype=x.dtype)
        return x * (1.0 + self.strength * torch.tanh(gain)) + self.strength * (torch.tanh(bias) + ray_term)
