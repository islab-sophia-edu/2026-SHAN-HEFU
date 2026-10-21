"""Stanford panorama world-normal to PanoMAE ERP normal coordinates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ERP_TO_POSE_BASIS = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def pose_rotation_column(euler_xyz):
    """Stanford final_camera_rotation radians; Rz(z) @ Ry(y) @ Rx(x)."""
    x, y, z = map(float, euler_xyz)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


def pose_path_from_normal(normal_path):
    path = Path(normal_path)
    for suffix in ("_normals", "_normal"):
        if path.stem.endswith(suffix):
            stem = path.stem[:-len(suffix)]
            target = path.parent.parent / "pose" / (stem + "_pose.json")
            if not target.is_file():
                raise FileNotFoundError(f"Missing Stanford panorama pose: {target}")
            return target
    raise ValueError(f"Unexpected Stanford normal filename: {path.name}")


def erp_to_world_rotation_row(pose_path):
    """R such that [x_erp, y_erp, z_erp] @ R = [x_world, y_world, z_world]."""
    with Path(pose_path).open('r', encoding='utf-8') as stream:
        data = json.load(stream)
    rotation = data.get('final_camera_rotation')
    if not isinstance(rotation, (tuple, list)) or len(rotation) != 3:
        raise ValueError(f"Pose missing final_camera_rotation: {pose_path}")
    matrix = ERP_TO_POSE_BASIS @ pose_rotation_column(rotation).T
    if abs(np.linalg.det(matrix) - 1.) > 1e-5:
        raise ValueError(f"Invalid camera pose rotation: {pose_path}")
    return matrix


def world_normals_to_erp(normal_world, pose_path):
    """Apply inverse camera-to-world rotation to row-vector normals; preserve zeros."""
    normal_world = np.asarray(normal_world, dtype=np.float32)
    if normal_world.shape[-1] != 3:
        raise ValueError(f"Normal last dimension must be 3, got {normal_world.shape}")
    rotation = erp_to_world_rotation_row(pose_path).astype(np.float32)
    return normal_world @ rotation.T


def erp_normals_to_world(normal_erp, pose_path):
    """Inverse of world_normals_to_erp, for world-frame reporting only."""
    rotation = erp_to_world_rotation_row(pose_path).astype(np.float32)
    return np.asarray(normal_erp, dtype=np.float32) @ rotation
