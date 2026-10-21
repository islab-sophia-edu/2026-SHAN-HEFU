#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'Segmentation'), str(ROOT)]

from main_finetune_segmentation_gctt import get_args_parser
from datasets_segmentation_8class_gctt import build_segmentation_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--samples', type=int, default=5)
    cli = p.parse_args()

    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    print('Checkpoint keys:', list(checkpoint))
    state = checkpoint.get('model', checkpoint)
    for key, weight in state.items():
        if key.endswith(('head.weight', 'decoder_pred.weight', 'seg_head.weight')):
            print('Classifier weight:', key, tuple(weight.shape))
    saved = checkpoint.get('args')
    if saved is not None:
        for key in ('model', 'pano_h', 'pano_w', 'grid_height', 'nb_classes', 'in_chans', 'val_data_path'):
            print('Saved', key, '=', getattr(saved, key, 'missing') if not isinstance(saved, dict) else saved.get(key, 'missing'))

    args = get_args_parser().parse_args([])
    args.data_path = cli.data_path
    args.val_data_path = cli.data_path
    args.pano_h, args.pano_w = 832, 1664
    args.grid_height = 16
    args.nb_classes = 8
    args.aug_device = 'cpu'
    args.input_size = (
        args.pano_h // args.grid_height,
        args.pano_w // (2 * args.grid_height),
    )
    dataset = build_segmentation_dataset(is_train=False, args=args)
    print('Val images:', len(dataset))
    summary = Counter()
    for i in range(min(cli.samples, len(dataset))):
        views, angles, gauge, target = dataset[i]
        values, counts = torch.unique(target, return_counts=True)
        summary.update({int(k): int(v) for k, v in zip(values, counts)})
        print(f'[{i}] views={tuple(views.shape)}, mean={views.float().mean():.4f}, std={views.float().std():.4f}, labels={dict(zip(values.tolist(), counts.tolist()))}')
        if not torch.isfinite(views).all():
            raise RuntimeError(f'Nonfinite RGB at index {i}')
    print('Aggregate label counts:', dict(sorted(summary.items())))
    if sum(v for k, v in summary.items() if k != 0) == 0:
        raise RuntimeError('All sampled labels are zero. Check label decoding and dataset exceptions.')


if __name__ == '__main__':
    main()
