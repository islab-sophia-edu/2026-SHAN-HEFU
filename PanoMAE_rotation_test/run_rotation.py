#!/usr/bin/env python3
"""Repository-native rotation evaluation launcher for PanoMAE.

Runs the original downstream --eval code while rotating validation ERP assets
before the task-specific dataset performs tangent extraction. No fine-tuning.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from so3_rotation import ERPSO3Rotator

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ('gctt', 'pb', '3dpe', '2dpe', 'grid', 'noadaptive', 'vanilla')
ANGLES = {
    'yaw': tuple(range(0, 360, 15)),
    'pitch': tuple(range(-90, 91, 15)),
    'roll': tuple(range(-180, 181, 30)),
}


def arguments():
    p = argparse.ArgumentParser(description='PanoMAE repository-native rotation evaluation')
    p.add_argument('--task', required=True, choices=('classification', 'segmentation', 'depth', 'normal'))
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--grid_height', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--num_classes', type=int, default=None,
                   help='Semantic classes, NOT number of workers; auto-detected for segmentation')
    p.add_argument('--dataset', choices=('auto', 'cvrg', 'stanford'), default='auto')
    p.add_argument('--model', default='vit_huge_patch14')
    p.add_argument('--pano_h', type=int)
    p.add_argument('--pano_w', type=int)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--axes', nargs='+', choices=tuple(ANGLES), default=list(ANGLES))
    p.add_argument('--angles', nargs='+', type=float,
                   help='Override the default degrees on every selected axis')
    p.add_argument('--yaw_step', type=int, default=15, help='Yaw interval in degrees')
    p.add_argument('--pitch_step', type=int, default=15, help='Pitch interval in degrees')
    p.add_argument('--roll_step', type=int, default=30, help='Roll interval in degrees')
    p.add_argument('--max_images', type=int, default=None)
    p.add_argument('--normal_label_source', choices=('normal', 'depth'), default=None)
    p.add_argument('--normal_target_frame', choices=('global', 'local'), default=None)
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def resolve_module(task, variant, dataset):
    folder = {'classification': 'Classification', 'segmentation': 'Segmentation',
              'depth': 'Depth', 'normal': 'Normal'}[task]
    if task == 'classification':
        pattern = 'main_PanoMAE_classification_{}'
    elif task == 'segmentation':
        pattern = ('main_stanford_segmentation_{}' if dataset == 'stanford'
                   else 'main_finetune_segmentation_{}')
    elif task == 'depth':
        pattern = 'main_finetune_depth_{}'
    else:
        pattern = 'main_finetune_normal_{}'
    # Vanilla MAE has repository-dependent fine-tune runners; do not silently load
    # a GCTT architecture with a vanilla checkpoint.
    if variant == 'vanilla':
        raise RuntimeError('Vanilla MAE requires its own downstream evaluation adapter; '
                           'no matching universal vanilla main_* file was verified.')
    name = pattern.format(variant)
    if not (ROOT / folder / (name + '.py')).is_file():
        raise FileNotFoundError(f'No task runner found: {ROOT / folder / (name + ".py")}')
    return folder, name


def get_parser(module):
    parser_fn = getattr(module, "get_args_parser", None)
    if callable(parser_fn):
        return parser_fn()

    module_path = getattr(module, "__file__", "unknown")
    raise RuntimeError(
        f"Module {module.__name__} does not provide get_args_parser(). "
        f"Loaded from: {module_path}"
    )


def detect_dataset(cli):
    if cli.task != 'segmentation':
        return 'stanford' if cli.task in ('depth', 'normal') else 'classification'
    if cli.dataset != 'auto':
        return cli.dataset
    path = Path(cli.data_path)
    return 'stanford' if ('stanford' in path.as_posix().lower() or
                          any((path / f'area_{s}').exists() for s in ('5a', '5b'))) else 'cvrg'


def rotate_pil(source, path, rotator, angles):
    if not any(angles.values()):
        return source
    path = Path(str(path)).as_posix().lower()
    name = Path(path).parent.name.lower()
    # RGB masks (e.g. Stanford anno) must be spatially warped with NEAREST,
    # preserving the exact palette; depth must never be quantized through RGB.
    if name in ('normal', 'normals'):
        source.close()
        raise RuntimeError('Nonzero normal-GT rotation requires a verified normal-frame '
                           'conversion. Refusing to report incorrect angular metrics.')
    is_discrete = name in ('mask', 'masks', 'anno', 'label', 'labels', 'depth')
    arr = np.asarray(source)
    if arr.ndim not in (2, 3):
        raise ValueError(f'Unexpected source image shape {arr.shape}: {path}')
    dtype = arr.dtype
    if np.issubdtype(dtype, np.integer):
        src = np.asarray(arr, dtype=np.float32)
    else:
        src = arr.astype(np.float32, copy=False)
    ten = torch.from_numpy(np.ascontiguousarray(src))
    if ten.ndim == 2:
        ten = ten[None, None]
    else:
        ten = ten.permute(2, 0, 1).unsqueeze(0)
    rot = rotator.rotate(ten, **angles, mode='nearest' if is_discrete else 'bilinear')
    out = rot.squeeze(0).permute(1, 2, 0).cpu().numpy() if arr.ndim == 3 else rot[0, 0].cpu().numpy()
    if np.issubdtype(dtype, np.integer):
        out = np.rint(out).clip(np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        out = out.astype(dtype)
    source.close()
    return Image.fromarray(out)


class RotatedValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, axis, degree):
        self.dataset = dataset
        self.axis = axis
        self.degree = degree
        self.angles = dict(yaw=0.0, pitch=0.0, roll=0.0)
        self.angles[axis] = float(degree)
        self.rotator = ERPSO3Rotator()
        self.original_open = Image.open

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        def rotating_open(fp, *args, **kwargs):
            img = self.original_open(fp, *args, **kwargs)
            if isinstance(fp, (str, os.PathLike)):
                return rotate_pil(img, fp, self.rotator, self.angles)
            # Classification uses a binary file handle: derive its name.
            if hasattr(fp, 'name'):
                return rotate_pil(img, fp.name, self.rotator, self.angles)
            return img
        with patch.object(Image, 'open', side_effect=rotating_open):
            return self.dataset[index]


def main():
    cli = arguments()
    if cli.num_workers:
        print('[note] For safe per-sample ERP interception, enforcing num_workers=0.')
    dataset = detect_dataset(cli)
    folder, module_name = resolve_module(cli.task, cli.variant, dataset)
    ckpt = Path(cli.checkpoint)
    if not ckpt.is_file() and not cli.dry_run:
        raise FileNotFoundError(ckpt)
    if not Path(cli.data_path).is_dir() and not cli.dry_run:
        raise FileNotFoundError(cli.data_path)
    step_sizes = {'yaw': cli.yaw_step, 'pitch': cli.pitch_step, 'roll': cli.roll_step}
    if any(step <= 0 for step in step_sizes.values()):
        raise ValueError('Rotation angle steps must be positive')
    degrees = {'yaw': tuple(range(0, 360, cli.yaw_step)),
               'pitch': tuple(range(-90, 91, cli.pitch_step)),
               'roll': tuple(range(-180, 181, cli.roll_step))}
    if cli.task == 'normal' and cli.angles is None:
        cli.axes = ['yaw']
        cli.angles = [0.0]
        print('[normal] Official normal PNG: zero-rotation validation only.')
    if cli.task == 'normal' and any(float(x) != 0 for axis in cli.axes for x in (cli.angles if cli.angles is not None else degrees[axis])):
        # Standalone evaluation engine must rotate both normal-map pixels and
        # vector components, potentially using pose-aware per-image transforms.
        raise RuntimeError('Normal rotation is not enabled until official Stanford '
                           'normal-GT frame conversion is integrated and validated.')
    sys.path.insert(0, str(ROOT / folder))
    sys.path.insert(1, str(ROOT))
    module = importlib.import_module(module_name)
    parser = get_parser(module)
    args = parser.parse_args([])
    args.grid_height = cli.grid_height
    args.model = cli.model
    args.batch_size = cli.batch_size
    args.num_workers = 0
    args.device = cli.device
    args.eval = True
    args.resume = str(ckpt)
    args.finetune = ''
    args.val_data_path = str(cli.data_path)
    args.output_dir = str(cli.output_dir)
    args.log_dir = str(cli.output_dir)
    if cli.task == 'classification':
        args.data_path = str(Path(cli.data_path).parent / 'train')
        if not Path(args.data_path).is_dir() and not cli.dry_run:
            raise FileNotFoundError('Classification needs the original train/ class mapping: ' + args.data_path)
    elif dataset == 'cvrg':
        args.data_path = str(Path(cli.data_path).parent / 'train')
    else:
        args.data_path = str(cli.data_path)
    if hasattr(args, 'nb_classes'):
        if cli.task == 'segmentation':
            args.nb_classes = cli.num_classes if cli.num_classes is not None else (13 if dataset == 'stanford' else 8)
        elif cli.task == 'classification':
            # Classification runner automatically sets class count from train mapping.
            if cli.num_classes is not None:
                print('[note] Ignoring --num_classes for classification; class mapping is authoritative.')
    if cli.pano_h is not None:
        args.pano_h = cli.pano_h
    if cli.pano_w is not None:
        args.pano_w = cli.pano_w
    if cli.task == 'classification' and cli.pano_h is None:
        args.pano_h, args.pano_w = 512, 1024
    elif cli.task == 'segmentation' and dataset == 'cvrg' and cli.pano_h is None:
        args.pano_h, args.pano_w = 832, 1664
    elif cli.task in ('segmentation', 'depth', 'normal') and cli.pano_h is None:
        args.pano_h, args.pano_w = 1024, 2048
    if cli.max_images is not None and hasattr(args, 'debug_limit'):
        args.debug_limit = cli.max_images
    if cli.task == 'normal':
        if cli.normal_label_source == 'depth':
            raise ValueError('This runner requires official Stanford normal PNG; --normal_label_source depth is disabled.')
        args.normal_label_source = 'normal'
        checkpoint_meta = torch.load(ckpt, map_location='cpu', weights_only=False) if not cli.dry_run else {}
        saved_args = checkpoint_meta.get('args', {})
        if not isinstance(saved_args, dict):
            saved_args = vars(saved_args) if hasattr(saved_args, '__dict__') else {}
        for name in (
            'normal_target_frame', 'normal_axis_perm', 'normal_axis_sign',
            'normal_local_y_sign', 'invalid_normal_value',
            'invalid_normal_tolerance', 'in_chans', 'ray_coord',
            'use_ray_token', 'use_encoder_adapters', 'use_aux_normal',
            'depth_encoder_depth', 'stage_encoder_depth', 'cnn_base_dim',
        ):
            if name in saved_args and hasattr(args, name):
                setattr(args, name, saved_args[name])
        if cli.normal_target_frame is not None:
            args.normal_target_frame = cli.normal_target_frame
        print(f'[normal] GT=official PNG | target_frame={args.normal_target_frame}')
        del checkpoint_meta
    if getattr(args, 'aug_device', '') == 'cuda':
        args.aug_device = 'cpu'
    out = Path(cli.output_dir)
    configs = [(axis, x) for axis in cli.axes for x in (cli.angles if cli.angles is not None else degrees[axis])]
    config = {'task': cli.task, 'variant': cli.variant, 'dataset': dataset,
              'checkpoint': str(ckpt), 'data_path': cli.data_path,
              'model': cli.model, 'pano_h': args.pano_h, 'pano_w': args.pano_w,
              'num_classes': getattr(args, 'nb_classes', None), 'num_workers': 0,
              'rotations': configs}
    print(json.dumps(config, indent=2))
    if cli.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    # Single concise report replaces per-angle logs and protocol files.
    # Wrap only validation dataset; training dataset remains untouched.
    function_name = ('build_classification_dataset' if cli.task == 'classification' else
                     'build_segmentation_dataset' if cli.task == 'segmentation' else
                     'build_depth_dataset' if cli.task == 'depth' else 'build_normal_dataset')
    original = getattr(module, function_name)
    results = []
    primary = {'classification': ('acc1', True),
               'segmentation': ('miou', True),
               'depth': ('abs_rel', False),
               'normal': ('mean', False)}[cli.task]

    def extract_metrics(text):
        patterns = {
            'classification': {
                'acc1': r'\* Acc@1\s+([\d.]+)',
                'acc5': r'Acc@5\s+([\d.]+)',
                'loss': r'\* Acc@1[^\n]*?loss\s+([\d.]+)',
            },
            'segmentation': {
                'miou': r'Validation mIoU:\s*([\d.]+)',
                'macc': r'mAcc:\s*([\d.]+)',
            },
            'depth': {
                'rmse': r'Validation RMSE:\s*([\d.]+)',
                'abs_rel': r'AbsRel:\s*([\d.]+)',
                'delta1': r'Delta1:\s*([\d.]+)',
            },
            'normal': {
                'mean': r'Val Normal:\s*Mean\s+([\d.]+)',
                'median': r'Val Normal:[^\n]*?Median\s+([\d.]+)',
                'rmse': r'Val Normal:[^\n]*?RMSE\s+([\d.]+)',
                'p11': r'Val Normal:[^\n]*?P11\s+([\d.]+)',
                'p22': r'Val Normal:[^\n]*?P22\s+([\d.]+)',
                'p30': r'Val Normal:[^\n]*?P30\s+([\d.]+)',
            },
        }
        metrics = {}
        for key, pat in patterns[cli.task].items():
            found = re.findall(pat, text)
            if found:
                metrics[key] = float(found[-1])
        if primary[0] not in metrics:
            tail = '\n'.join(text.splitlines()[-30:])
            raise RuntimeError('Could not extract primary evaluation metric. Last output:\n' + tail)
        return metrics

    def write_summary():
        metric, higher = primary
        valid = [r for r in results if metric in r['metrics']]
        rank = (lambda r: r['metrics'][metric])
        best = (max(valid, key=rank) if higher else min(valid, key=rank)) if valid else None
        worst = (min(valid, key=rank) if higher else max(valid, key=rank)) if valid else None
        keys = list(dict.fromkeys(k for r in results for k in r['metrics']))

        def fmt(r):
            return ', '.join(f'{k}={r["metrics"][k]:.4f}' for k in keys if k in r['metrics'])

        def mean_stats(rows):
            return ', '.join(
                f'{k}={np.mean([r["metrics"][k] for r in rows if k in r["metrics"]]):.4f}'
                for k in keys if any(k in r['metrics'] for r in rows)
            )

        lines = [
            'PanoMAE Rotation Validation',
            f'Task: {cli.task} | Variant: {cli.variant} | Dataset: {dataset}',
            f'Checkpoint: {ckpt}',
            f'Data: {cli.data_path}',
            f'Completed: {len(results)}/{len(configs)}',
            f'Primary metric: {metric} ({"higher" if higher else "lower"} is better)',
            '', '[All evaluations]',
        ]
        for i, r in enumerate(results, 1):
            lines.append(f'{i:02d}. {r["axis"]:5s} {r["degree"]:7g} deg | {fmt(r)}')
        lines += ['', '[Overall validation mean]', mean_stats(results) if results else 'Pending']
        lines += ['', '[Axis-wise validation mean]']
        for axis in cli.axes:
            rows = [r for r in results if r['axis'] == axis]
            if rows:
                lines.append(f'{axis}: n={len(rows)} | {mean_stats(rows)}')
        if best is not None:
            lines += [
                '', '[Best rotation]',
                f'{best["axis"]} {best["degree"]:g} deg | {fmt(best)}',
                '', '[Worst rotation]',
                f'{worst["axis"]} {worst["degree"]:g} deg | {fmt(worst)}',
            ]
            baseline = next((r for r in valid if r['degree'] == 0), None)
            if baseline is not None:
                lines += ['', '[Zero rotation baseline]', fmt(baseline)]
                base = baseline['metrics'][metric]
                if base != 0:
                    lines.append(f'Overall primary-metric retention: '
                                 f'{np.mean([r["metrics"][metric] for r in valid]) / base:.4f}')
        lines += ['', 'Note: Best and worst are among tested angles; overall mean is the unweighted mean over test configurations.']
        path = out / 'val.txt'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return best

    for index, (axis, degree) in enumerate(configs, start=1):
        label = f'{axis} {degree:g} deg'
        current_args = parser.parse_args([])
        current_args.__dict__.update(args.__dict__)
        current_args.output_dir = str(out)
        current_args.log_dir = str(out)

        def replacement(*a, **kw):
            ds = original(*a, **kw)
            train = kw.get('is_train', a[0] if a else None)
            return ds if train else RotatedValDataset(ds, axis, degree)

        print(f'[{index}/{len(configs)}] {label} ...', flush=True)
        capture = io.StringIO()
        try:
            with patch.object(module, function_name, side_effect=replacement):
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                    module.main(current_args)
            metrics = extract_metrics(capture.getvalue())
        except Exception:
            # Preserve diagnostics for a failed run, without creating per-angle logs.
            print(capture.getvalue().splitlines()[-30:], file=sys.stderr)
            raise
        finally:
            capture.close()
        results.append({'axis': axis, 'degree': degree, 'metrics': metrics})
        print('    ' + ', '.join(f'{k}={v:.4f}' for k,v in metrics.items()), flush=True)
        write_summary()

    best = write_summary()
    if best is not None:
        print(f"\nBEST: {best['axis']} {best['degree']:g} deg | {primary[0]}={best['metrics'][primary[0]]:.4f}")
    print(f'One result file: {out / "val.txt"}')
    print('Verify 0-degree results match the original validation protocol.')


if __name__ == '__main__':
    main()
