import math
import sys
import torch
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, max_norm=0, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    # 明确添加 acc1 和 acc5 的 meter
    metric_logger.add_meter('acc1', misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
    
    header = 'Epoch: [{}]'.format(epoch)

    for data_iter_step, (views, angles, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if data_iter_step % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # GPU Transfer
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(views, angles)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        # 计算 Acc (用于监控和 log)
        torch.cuda.synchronize()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        batch_size = views.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # 同步所有进程的统计结果
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # [CRITICAL FIX] 确保所有 key 都在这里返回
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()

    for views, angles, target in metric_logger.log_every(data_loader, 10, 'Test:'):
        views = views.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(views, angles)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = views.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}