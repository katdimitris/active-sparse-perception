# --------------------------------------------------------
# Based on Swin Transformer (Copyright (c) 2021 Microsoft, MIT License)
# --------------------------------------------------------
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """Linear warmup, then cosine decay; updated every iteration."""
    return CosineLRScheduler(
        optimizer,
        t_initial=int(config.TRAIN.EPOCHS * n_iter_per_epoch),
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch),
        cycle_limit=1,
        t_in_epochs=False,
    )
