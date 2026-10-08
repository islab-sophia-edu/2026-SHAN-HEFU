import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy

import util.lr_sched as lr_sched
import util.misc as misc


def _unpack_batch(batch):
    """
    Backward-compatible classification batch unpacking.

    Old classification dataset:
        views, angles, targets

    GCTT classification dataset:
        views, angles, gauge_angles, targets
    """
    if len(batch) == 4:
        views, angles, gauge_angles, targets = batch
    elif len(batch) == 3:
        views, angles, targets = batch
        gauge_angles = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}. Expected 3 or 4.")
    return views, angles, gauge_angles, targets


def _move_to_device(views, angles, gauge_angles, targets, device):
    views = views.to(device, non_blocking=True)
    angles = angles.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if gauge_angles is not None:
        gauge_angles = gauge_angles.to(device, non_blocking=True)
    return views, angles, gauge_angles, targets


def _autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type=device.type, enabled=False)


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
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.8f}"))

    header = f"Epoch: [{epoch}]"
    print_freq = 10
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args,
            )

        views, angles, gauge_angles, targets = _unpack_batch(batch)
        views, angles, gauge_angles, targets = _move_to_device(
            views,
            angles,
            gauge_angles,
            targets,
            device,
        )

        if mixup_fn is not None:
            # Mixup changes only image tensors and labels.
            # angles/gauge_angles remain the geometry attached to each sample.
            views, targets = mixup_fn(views, targets)

        with _autocast_context(device):
            outputs = model(views, angles, gauge_angles=gauge_angles)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter

        is_last_batch = data_iter_step == len(data_loader) - 1
        is_update_step = ((data_iter_step + 1) % accum_iter == 0) or is_last_batch

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
                log_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch_1000x)
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

    for batch in metric_logger.log_every(data_loader, 10, header):
        views, angles, gauge_angles, target = _unpack_batch(batch)
        views, angles, gauge_angles, target = _move_to_device(
            views,
            angles,
            gauge_angles,
            target,
            device,
        )

        with _autocast_context(device):
            output = model(views, angles, gauge_angles=gauge_angles)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = views.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
