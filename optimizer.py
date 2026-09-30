# --------------------------------------------------------
# Based on Swin Transformer (Copyright (c) 2021 Microsoft, MIT License)
# --------------------------------------------------------
import torch.nn as nn
from torch import optim


def build_optimizer(config, model):
    """AdamW; no weight decay on normalisation layers, biases, 1-d parameters and the
    parameters the model lists in no_weight_decay() (matched as name prefixes)."""
    skip_prefixes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            skip_prefixes.append(name)
        elif hasattr(module, 'no_weight_decay'):
            skip_prefixes += [f'{name}.{p}' if name else p for p in module.no_weight_decay()]
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith('.bias') or any(name.startswith(p) for p in skip_prefixes if p):
            no_decay.append(param)
        else:
            decay.append(param)
    return optim.AdamW([{'params': decay}, {'params': no_decay, 'weight_decay': 0.}],
                       lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                       eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS)
