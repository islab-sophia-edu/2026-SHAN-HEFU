"""Export the sampling camera rotation for every panorama patch.

The export is a diagnostic of the known patch sampling geometry, not a learned
camera pose estimator and not Stanford pano/pose JSON.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from decoder_camera_pose import DecoderCameraPose


def main():
    parser = argparse.ArgumentParser(description="Export ODI patch-facing camera frames")
    parser.add_argument("--grid_height", type=int, default=16)
    parser.add_argument("--gauge_deg", type=float, default=0.0)
    parser.add_argument("--grid", action="store_true", help="Use ERP grid cell poses instead of tangent cameras")
    parser.add_argument("--save_ray_resolution", type=int, default=0,
                        help="Ray subpixels per patch; 0 saves only the per-patch rotations")
    parser.add_argument("--output_dir", type=Path, default=Path("./patch_camera_poses"))
    args = parser.parse_args()
    gh, gw = args.grid_height, 2 * args.grid_height
    if args.grid:
        row = torch.arange(gh, dtype=torch.float32)
        col = torch.arange(gw, dtype=torch.float32)
        lat = (0.5 - (row + .5) / gh) * 180
        lon = ((col + .5) / gw * 2 - 1) * 180
    else:
        lat = torch.linspace(90 - 90 / gh, -90 + 90 / gh, gh)
        lon = torch.linspace(-180, 180, gw + 1)[:-1]
    phi, theta = torch.meshgrid(lat, lon, indexing="ij")
    angles = torch.stack((theta, phi), dim=-1).reshape(1, -1, 2)
    gauge = torch.full((1, gh * gw, 1), args.gauge_deg) if not args.grid else None
    decoder_pose = DecoderCameraPose(grid_height=gh, is_grid=args.grid)
    forward, right, up = decoder_pose._frames(angles, gauge)
    rotation = decoder_pose.camera_rotation(angles, gauge)
    err = (rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max()
    det = torch.linalg.det(rotation)
    if err > 1e-5 or not torch.allclose(det, torch.ones_like(det), atol=1e-5):
        raise RuntimeError(f"Invalid camera rotations: orthogonality={err}, determinants={det.min()}..{det.max()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = dict(
        angles_deg=angles[0].numpy(),
        gauge_deg=np.full((gh * gw,), 0. if args.grid else args.gauge_deg, dtype=np.float32),
        forward=forward[0].numpy(), right=right[0].numpy(), up=up[0].numpy(),
        R_cam_to_ERP=rotation[0].numpy(),
    )
    if args.save_ray_resolution:
        data['rays'] = decoder_pose.patch_rays(angles, gauge, args.save_ray_resolution)[0].numpy()
    np.savez_compressed(args.output_dir / "patch_camera_pose.npz", **data)
    with (args.output_dir / "patch_camera_pose.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("patch", "row", "col", "longitude_deg", "latitude_deg", "gauge_deg",
                         "forward_x", "forward_y", "forward_z", "right_x", "right_y", "right_z",
                         "up_x", "up_y", "up_z"))
        a = data["angles_deg"]
        for idx in range(gh * gw):
            writer.writerow((idx, idx // gw, idx % gw, *a[idx].tolist(), data['gauge_deg'][idx],
                             *data['forward'][idx].tolist(), *data['right'][idx].tolist(),
                             *data['up'][idx].tolist()))
    print(f"Saved {gh * gw} patch poses to {args.output_dir}")
    print("First patch: lon/lat=", data["angles_deg"][0], "forward=", data["forward"][0])
    print(f"R_cam_to_ERP: det in [{det.min().item():.6f}, {det.max().item():.6f}]")


if __name__ == "__main__":
    main()
