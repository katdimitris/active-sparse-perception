import os

import torch
import torch.distributed as dist


def reduce_mean(tensor):
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


def save_checkpoint(path, **state):
    """Write to a temporary file and rename, so an interrupted save never leaves a
    truncated checkpoint behind."""
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def latest_checkpoint(output_dir):
    path = os.path.join(output_dir, 'checkpoint.pth')
    return path if os.path.exists(path) else None


def load_checkpoint(path, model, optimizer=None, lr_scheduler=None, scaler=None, logger=None):
    """Returns the epoch to resume from and the best accuracy so far."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(ckpt['model'])
    if logger:
        logger.info(f'loaded {path}: {msg}')
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        return ckpt['epoch'] + 1, ckpt.get('max_accuracy', 0.)
    return 0, ckpt.get('max_accuracy', 0.)
