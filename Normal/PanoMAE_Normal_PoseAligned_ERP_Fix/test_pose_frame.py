"""Tests Pose-aligned GCTT normal conventions and provided v2 geometry audit."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Normal.normal_pose_frame import (
    ERP_TO_POSE_BASIS, erp_to_world_rotation_row,
    erp_normals_to_world, world_normals_to_erp,
    pose_path_from_normal,
)


def rotation_error_degrees(a, b):
    value = (np.trace(a @ b.T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(value, -1., 1.))))


def test_synthetic():
    assert np.linalg.det(ERP_TO_POSE_BASIS) == 1.0
    rng = np.random.default_rng(9)
    with TemporaryDirectory() as directory:
        base = Path(directory)
        (base / 'normal').mkdir()
        (base / 'pose').mkdir()
        normal_path = base / 'normal' / 'camera_test_normals.png'
        pose_path = base / 'pose' / 'camera_test_pose.json'
        for angles in [(0., 0., 0.), (1.5, 0.02, -1.9), (-1.55, -0.015, 3.12)]:
            pose_path.write_text(json.dumps({'final_camera_rotation': angles}))
            assert pose_path_from_normal(normal_path) == pose_path
            n_world = rng.normal(size=(6, 7, 3)).astype(np.float32)
            n_world /= np.linalg.norm(n_world, axis=-1, keepdims=True)
            n_world[2, 3] = 0
            n_erp = world_normals_to_erp(n_world, pose_path)
            recovered = erp_normals_to_world(n_erp, pose_path)
            assert np.max(np.abs(recovered - n_world)) < 2e-7
            rot = erp_to_world_rotation_row(pose_path)
            assert np.linalg.norm(rot @ rot.T - np.eye(3)) < 1e-12
            assert abs(np.linalg.det(rot)-1) < 1e-12


def test_real_audit(path):
    with open(path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    errors = []
    from Normal.normal_pose_frame import pose_rotation_column
    for row in rows:
        fitted = np.array(json.loads(row['erp_to_world_rotation_json']))
        pose_rotation = pose_rotation_column(json.loads(row['final_camera_rotation_json']))
        calculated = ERP_TO_POSE_BASIS @ pose_rotation.T
        errors.append(rotation_error_degrees(calculated, fitted))
    assert max(errors) < 0.02, max(errors)
    area6 = [e for row, e in zip(rows, errors) if row['area']=='area_6']
    assert len(area6) == 8
    assert max(area6) < 0.02
    print(f'PASS: {len(rows)} pose comparisons; residual median={np.median(errors):.6f}deg max={max(errors):.6f}deg')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path)
    args = p.parse_args()
    test_synthetic()
    print('PASS: handedness, unit rotation, and synthetic bidirectional normal roundtrip')
    if args.csv:
        test_real_audit(args.csv)
