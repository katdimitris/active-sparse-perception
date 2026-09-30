"""Train or evaluate a recurrent (or static) SparseFormer on an ImageFolder dataset.

    torchrun --nproc_per_node=2 train.py --cfg configs/imagenet100/recurrent_9-16-25_memory.yaml \
        --data-path /data/imagenet100 --output output

Every round is supervised (deep supervision): cross-entropy on the last exit plus
SF.W_EARLY times the mean cross-entropy of the earlier exits.
"""
import argparse
import datetime
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import latest_checkpoint, load_checkpoint, reduce_mean, save_checkpoint


def parse_option():
    parser = argparse.ArgumentParser('Recurrent SparseFormer training and evaluation')
    parser.add_argument('--cfg', required=True, help='path to a config file')
    parser.add_argument('--opts', nargs='+', default=None, help='override config entries: KEY VALUE ...')
    parser.add_argument('--batch-size', type=int, help='batch size per GPU')
    parser.add_argument('--data-path', help='ImageFolder root with train/ and val/')
    parser.add_argument('--resume', help='checkpoint to resume from (or to evaluate with --eval)')
    parser.add_argument('--eval', action='store_true', help='only evaluate --resume')
    parser.add_argument('--output', default='output')
    parser.add_argument('--tag', help='run name (default: the config file name)')
    args = parser.parse_args()
    return args, get_config(args)


def deep_supervision_loss(logits, targets, criterion, w_early):
    loss = criterion(logits[-1], targets)
    if len(logits) > 1 and w_early > 0:
        loss = loss + w_early * torch.stack([criterion(z, targets) for z in logits[:-1]]).mean()
    return loss


def gflops_per_round(model, img_size):
    """Cost of stopping after round 1, 2, ... (fvcore counts one multiply-add as one FLOP)."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    x, cost = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device), []
    was_training = model.training
    model.eval()
    for r in range(1, model.num_rounds + 1):
        model.max_round = r
        f = FlopCountAnalysis(model, x)
        f.unsupported_ops_warnings(False)
        f.uncalled_modules_warnings(False)
        cost.append(f.total() / 1e9)
    model.max_round = None
    model.train(was_training)
    return cost


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                    scaler, logger):
    model.train()
    num_steps = len(data_loader)
    batch_time, loss_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter()
    start = end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=config.AMP):
            # the loss scale is passed on so the model can clamp the sampling-point
            # gradients in unscaled units
            logits = model(samples, scale=scaler.get_scale() if config.AMP else 1.)
            loss = deep_supervision_loss(logits, targets, criterion, config.SF.W_EARLY)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            eta = datetime.timedelta(seconds=int(batch_time.avg * (num_steps - idx)))
            logger.info(f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  eta {eta}  '
                        f'lr {optimizer.param_groups[0]["lr"]:.6f}  time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  '
                        f'mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MB')
    loss_avg = reduce_mean(torch.tensor(loss_meter.avg, device='cuda')).item()
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(time.time() - start))}, '
                f'average loss {loss_avg:.4f}')
    return loss_avg


@torch.no_grad()
def validate(config, data_loader, model, logger):
    """Top-1 of every exit and top-5 of the last, counted exactly across ranks."""
    model.eval()
    correct, correct5, count = None, torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda')
    for images, target in data_loader:
        with torch.autocast('cuda', enabled=config.AMP):
            logits = model(images)
        if correct is None:
            correct = torch.zeros(len(logits), device='cuda')
        for r, z in enumerate(logits):
            correct[r] += (z.argmax(-1) == target).sum()
        correct5 += (logits[-1].topk(5, -1).indices == target[:, None]).any(-1).sum()
        count += target.numel()
    for t in (correct, correct5, count):
        dist.all_reduce(t)
    acc = (100 * correct / count).tolist()
    acc5 = (100 * correct5 / count).item()
    logger.info(f'Eval on {int(count.item())} images: top-1 per exit '
                + '  '.join(f'r{r + 1} {a:.2f}' for r, a in enumerate(acc)) + f'  | top-5 (last) {acc5:.2f}')
    return acc, acc5


def main(config, logger):
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{config.MODEL.NAME}: tokens per round {config.SF.TOKENS}, memory {config.SF.MEMORY}, '
                f'{n_params / 1e6:.2f}M parameters')
    if dist.get_rank() == 0:
        cost = gflops_per_round(model, config.DATA.IMG_SIZE)
        if cost:
            logger.info('GFLOPs when stopping after round r: '
                        + '  '.join(f'r{r + 1} {c:.3f}' for r, c in enumerate(cost)))

    optimizer = build_optimizer(config, model)
    model_ddp = DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()             # label smoothing is folded into the mixup targets
    elif config.MODEL.LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=config.AMP)

    start_epoch, max_accuracy = 0, 0.
    resume = config.MODEL.RESUME or (latest_checkpoint(config.OUTPUT) if config.TRAIN.AUTO_RESUME else None)
    if config.EVAL_MODE:
        assert resume, '--eval needs --resume'
        load_checkpoint(resume, model, logger=logger)
        validate(config, data_loader_val, model_ddp, logger)
        return
    if resume:
        start_epoch, max_accuracy = load_checkpoint(resume, model, optimizer, lr_scheduler, scaler, logger)
        logger.info(f'resuming from epoch {start_epoch}')

    tb = SummaryWriter(os.path.join(config.OUTPUT, 'tb')) if dist.get_rank() == 0 else None
    logger.info('Start training')
    start_time = time.time()
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(config, model_ddp, criterion, data_loader_train, optimizer, epoch,
                                     mixup_fn, lr_scheduler, scaler, logger)
        acc, acc5 = validate(config, data_loader_val, model_ddp, logger)
        max_accuracy = max(max_accuracy, acc[-1])
        last = epoch == config.TRAIN.EPOCHS - 1
        if tb is not None:
            tb.add_scalar('train/loss', train_loss, epoch)
            tb.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            for r, a in enumerate(acc):
                tb.add_scalar(f'val/top1_round{r + 1}', a, epoch)
            tb.add_scalar('val/top5', acc5, epoch)
            tb.flush()
        if dist.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or last):
            save_checkpoint(os.path.join(config.OUTPUT, 'checkpoint.pth'), model=model.state_dict(),
                            optimizer=optimizer.state_dict(), lr_scheduler=lr_scheduler.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, max_accuracy=max_accuracy,
                            config=config.dump())
        if dist.get_rank() == 0 and last:
            save_checkpoint(os.path.join(config.OUTPUT, 'final_weights.pth'), model=model.state_dict(),
                            epoch=epoch, config=config.dump())
        dist.barrier()
    if tb is not None:
        tb.close()
    logger.info(f'Training time {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == '__main__':
    args, config = parse_option()
    torch.cuda.set_device(config.LOCAL_RANK)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # learning rates in the configs are for a global batch of 512; scale linearly
    scale = config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    config.defrost()
    config.TRAIN.BASE_LR *= scale
    config.TRAIN.WARMUP_LR *= scale
    config.TRAIN.MIN_LR *= scale
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)
    if dist.get_rank() == 0:
        with open(os.path.join(config.OUTPUT, 'config.yaml'), 'w') as f:
            f.write(config.dump())
    logger.info(config.dump())
    main(config, logger)
