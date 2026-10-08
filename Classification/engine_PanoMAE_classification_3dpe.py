import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy

import util.lr_sched as lr_sched
import util.misc as misc


def _autocast_context(device: torch.device):
    """
    Use torch.amp.autocast with a safe device_type.
    """
    device_type = device.type
    enabled = device_type == "cuda"
    return torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=enabled,
    )


def _apply_mixup_to_pano_views(views, targets, mixup_fn):
    """
    timm Mixup/CutMix is written for [B, C, H, W].
    Pano views are [B, N, C, H, W].

    We reshape [B, N, C, H, W] -> [B, N*C, H, W], apply timm Mixup/CutMix,
    then reshape back. This preserves the batch-level mixing semantics and
    avoids CutMix slicing the wrong dimensions.
    """
    B, N, C, H, W = views.shape
    views_4d = views.reshape(B, N * C, H, W)
    views_4d, targets = mixup_fn(views_4d, targets)
    views = views_4d.reshape(B, N, C, H, W)
    return views, targets


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr",
        misc.SmoothedValue(window_size=1, fmt="{value:.8f}"),
    )

    header = f"Epoch: [{epoch}]"
    print_freq = 10
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (views, angles, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args,
            )

        is_last_batch = data_iter_step == len(data_loader) - 1
        is_update_step = ((data_iter_step + 1) % accum_iter == 0) or is_last_batch

        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            views, targets = _apply_mixup_to_pano_views(views, targets, mixup_fn)

        with _autocast_context(device):
            outputs = model(views, angles)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=is_update_step,
        )

        if is_update_step:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if log_writer is not None and is_update_step:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            try:
                log_writer.add_scalar("train/loss", loss_value, epoch_1000x)
                log_writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    epoch_1000x,
                )
            except Exception:
                pass

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    for views, angles, target in metric_logger.log_every(data_loader, 10, header):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with _autocast_context(device):
            output = model(views, angles)
            loss = criterion(output, target)

        maxk = min(5, output.shape[-1])
        if maxk >= 5:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc5_value = acc5.item()
        else:
            acc1 = accuracy(output, target, topk=(1,))[0]
            acc5_value = acc1.item()

        batch_size = views.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5_value, n=batch_size)

    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} "
        "loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
