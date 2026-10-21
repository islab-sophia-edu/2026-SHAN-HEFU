"""ERP SO(3) warping with the project's x=cos(lat)cos(lon), y=-cos(lat)sin(lon) convention."""
import math
import torch
import torch.nn.functional as F


def rotation_matrix(yaw, pitch, roll, *, device, dtype=torch.float32):
    y, p, r = [math.radians(float(x)) for x in (yaw, pitch, roll)]
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=device, dtype=dtype)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=device, dtype=dtype)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=device, dtype=dtype)
    return rz @ ry @ rx


class ERPSO3Rotator:
    """Active scene rotation R: output at d samples input at R.T @ d."""

    def __init__(self):
        self._cache = {}

    def _directions(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._cache:
            # Pixel centers; longitude's periodic seam lies between first/last columns.
            rows = torch.arange(h, device=device, dtype=torch.float32)
            cols = torch.arange(w, device=device, dtype=torch.float32)
            lat = math.pi * (0.5 - (rows + 0.5) / h)
            lon = 2 * math.pi * ((cols + 0.5) / w - 0.5)
            phi, theta = torch.meshgrid(lat, lon, indexing='ij')
            self._cache[key] = torch.stack([
                phi.cos() * theta.cos(), -phi.cos() * theta.sin(), phi.sin()
            ], -1).reshape(-1, 3)
        return self._cache[key]

    def rotate(self, image, yaw=0, pitch=0, roll=0, *, mode='bilinear', vector=False):
        if image.ndim != 4:
            raise ValueError('Expected BCHW tensor')
        if not image.is_floating_point():
            raise TypeError('Input must be floating point')
        if vector and image.shape[1] != 3:
            raise ValueError('Vector field must have 3 channels')
        if not any(float(a) for a in (yaw, pitch, roll)):
            return image.clone()
        b, c, h, w = image.shape
        R = rotation_matrix(yaw, pitch, roll, device=image.device)
        dirs = self._directions(h, w, image.device)
        src = dirs @ R  # row vectors: src = R.T @ dst
        lon = torch.atan2(-src[:, 1], src[:, 0])
        lat = torch.asin(src[:, 2].clamp(-1, 1))
        u = (lon / (2 * math.pi) + 0.5) * w - 0.5
        v = (0.5 - lat / math.pi) * h - 0.5
        # Add a seam halo so bilinear interpolation wraps horizontally.
        tiled = torch.cat((image[..., -1:], image, image[..., :1]), dim=-1)
        u = torch.remainder(u, w) + 1
        grid = torch.stack((2 * (u + 0.5) / (w + 2) - 1,
                            2 * (v + 0.5) / h - 1), -1).reshape(1, h, w, 2)
        out = F.grid_sample(tiled.float(), grid.expand(b, -1, -1, -1), mode=mode,
                            padding_mode='border', align_corners=False)
        if vector:
            # n_out = R @ n_in for active rotation in world coordinates.
            out = torch.einsum('ij,bjhw->bihw', R, out)
            out = F.normalize(out, dim=1, eps=1e-6)
        return out.to(image.dtype)

    def clear_cache(self):
        self._cache.clear()
