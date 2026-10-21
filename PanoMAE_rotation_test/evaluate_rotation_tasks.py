"""Task-agnostic rotation evaluation. Models accept rotated ERP tensors and return ERP predictions.

Adapters should implement preprocessing/tokenization and inverse ODI stitching INSIDE predict().
For classification, predict(image) -> logits [B,C].
For segmentation, predict(image) -> logits [B,C,H,W].
For depth, predict(image) -> depth [B,1,H,W] in meters.
For normal, predict(image) -> WORLD-frame normals [B,3,H,W].
Dataloader must return (rgb [B,3,H,W], target [B,...], valid [B,1,H,W] or None).
"""
import argparse
import csv
import torch
import torch.nn.functional as F
from so3_rotation import ERPSO3Rotator

ANGLES = {
    'yaw': [0, 45, 90, 135, 180, 225, 270, 315],
    'pitch': [-90, -60, -30, 0, 30, 60, 90],
    'roll': [-90, -60, -30, 0, 30, 60, 90],
}


def _valid_mask(valid, like):
    if valid is None:
        return torch.ones_like(like[:, :1], dtype=torch.bool)
    return (valid[:, :1] > 0.5) & torch.isfinite(valid[:, :1])


@torch.inference_mode()
def evaluate(task, dataloader, predict, device, *, num_classes=None, ignore_index=None,
             angles=None, output_csv='rotation_results.csv'):
    if task not in ('classification', 'segmentation', 'depth', 'normal'):
        raise ValueError(task)
    if task == 'segmentation' and not num_classes:
        raise ValueError('num_classes is required for segmentation')
    rotator = ERPSO3Rotator()
    rows = []
    for axis, axis_angles in (angles or ANGLES).items():
        for degree in axis_angles:
            cfg = {'yaw': 0, 'pitch': 0, 'roll': 0}
            cfg[axis] = degree
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device) if task == 'segmentation' else None
            total = torch.zeros(4, dtype=torch.float64, device=device)
            for batch in dataloader:
                rgb, target, valid = batch
                rgb = rgb.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True) if valid is not None else None
                image = rotator.rotate(rgb, **cfg)
                if task == 'classification':
                    logits = predict(image)
                    total[0] += (logits.argmax(1) == target.view(-1)).sum()
                    total[1] += target.numel()
                    continue
                if task == 'segmentation':
                    gt = rotator.rotate(target.unsqueeze(1).float() if target.ndim == 3 else target.float(),
                                        **cfg, mode='nearest')[:, 0].long()
                    logits = predict(image)
                    if logits.shape[-2:] != gt.shape[-2:]:
                        logits = F.interpolate(logits.float(), size=gt.shape[-2:], mode='bilinear', align_corners=False)
                    pred = logits.argmax(1)
                    mask = (gt >= 0) & (gt < num_classes)
                    if ignore_index is not None:
                        mask &= gt != ignore_index
                    if valid is not None:
                        mask &= rotator.rotate(valid.float(), **cfg, mode='nearest')[:, 0] > 0.5
                    idx = gt[mask] * num_classes + pred[mask]
                    confusion += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
                    continue
                gt = rotator.rotate(target.float(), **cfg, vector=task == 'normal')
                vm = rotator.rotate(valid.float(), **cfg, mode='nearest') if valid is not None else None
                pred = predict(image).float()
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                if task == 'depth':
                    mask = _valid_mask(vm, gt) & torch.isfinite(gt) & torch.isfinite(pred) & (gt > 1e-6)
                    err = (pred - gt).abs()
                    total[0] += (err / gt.clamp_min(1e-6))[mask].double().sum()
                    total[1] += err.square()[mask].double().sum()
                    total[2] += mask.sum()
                else:
                    mask = _valid_mask(vm, gt) & torch.isfinite(gt).all(1, keepdim=True) & torch.isfinite(pred).all(1, keepdim=True)
                    cosine = (F.normalize(pred, dim=1, eps=1e-6) * F.normalize(gt, dim=1, eps=1e-6)).sum(1, keepdim=True).clamp(-1, 1)
                    error_deg = torch.rad2deg(torch.acos(cosine))
                    total[0] += error_deg[mask].double().sum()
                    total[1] += (error_deg[mask] < 11.5).double().sum()
                    total[2] += mask.sum()
            if task == 'classification':
                result = {'acc1': (total[0] / total[1]).item() * 100 if total[1] else float('nan')}
            elif task == 'segmentation':
                tp = confusion.diag()
                union = confusion.sum(0) + confusion.sum(1) - tp
                eligible = union > 0
                result = {'miou': (tp[eligible] / union[eligible]).mean().item() * 100 if eligible.any() else float('nan')}
            elif task == 'depth':
                result = {'absrel': (total[0] / total[2]).item(), 'rmse': (total[1] / total[2]).sqrt().item()} if total[2] else {'absrel': float('nan'), 'rmse': float('nan')}
            else:
                result = {'mean_deg': (total[0] / total[2]).item(), 'delta_11_5': (total[1] / total[2]).item() * 100} if total[2] else {'mean_deg': float('nan'), 'delta_11_5': float('nan')}
            rows.append({'axis': axis, 'angle': degree, **result})
            print(rows[-1], flush=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
