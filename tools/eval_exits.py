"""Accuracy and cost of every exit of a trained model, and its early-exit curve.

The checkpoint is evaluated on its config's evaluation set on one GPU, with exact
counts.  The cost of stopping after round r is measured with fvcore at the config's
input size.  For early exit, an image stops at the first round whose top softmax
probability reaches a threshold t; sweeping t traces accuracy against the average
GFLOPs per image.  Everything goes to a JSON file that tools/plot_frontier.py reads.

    python tools/eval_exits.py --cfg configs/imagenet200/recurrent_9-16-25_memory.yaml \
        --ckpt output/recurrent_9-16-25_memory/final_weights.pth \
        --data-path /data/imagenet200 --out results/imagenet200/recurrent_9-16-25_memory.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import _C, _update_config_from_file  # noqa: E402
from data import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from train import gflops_per_round  # noqa: E402


def early_exit_curve(logits, labels, cost, n_points=400):
    """[(average GFLOPs, top-1 %)] over confidence thresholds, cheapest first."""
    conf = torch.stack([z.softmax(-1).max(-1).values for z in logits])       # [R, N]
    ok = torch.stack([z.argmax(-1).eq(labels) for z in logits])              # [R, N]
    cost = torch.tensor(cost)
    curve = []
    for t in np.linspace(float(conf.min()) - 1e-6, float(conf.max()) + 1e-6, n_points):
        exit_round = torch.full((conf.size(1),), len(logits) - 1, dtype=torch.long)
        for r in range(len(logits) - 2, -1, -1):
            exit_round = torch.where(conf[r] >= t, torch.full_like(exit_round, r), exit_round)
        curve.append((cost[exit_round].mean().item(),
                      100 * ok.gather(0, exit_round[None]).float().mean().item()))
    return sorted(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    config = _C.clone()
    _update_config_from_file(config, a.cfg)
    config.defrost()
    config.DATA.DATA_PATH = a.data_path
    dataset, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config)
    config.freeze()

    model = build_model(config)
    state = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state))
    model.eval()
    cost = gflops_per_round(model, config.DATA.IMG_SIZE)       # on CPU, before moving the model
    model.cuda()

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device='cuda').view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_DEFAULT_STD, device='cuda').view(1, 3, 1, 1) * 255
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.workers, pin_memory=True)
    logits, labels = None, []
    with torch.no_grad():
        for x, y in loader:
            x = (x.cuda(non_blocking=True).float() - mean) / std
            with torch.autocast('cuda'):
                out = model(x)
            logits = [[] for _ in out] if logits is None else logits
            for r, z in enumerate(out):
                logits[r].append(z.float().cpu())
            labels.append(y)
    logits = [torch.cat(z) for z in logits]
    labels = torch.cat(labels)
    top1 = [100 * z.argmax(-1).eq(labels).float().mean().item() for z in logits]

    result = {'name': config.MODEL.NAME, 'config': a.cfg, 'tokens': list(config.SF.TOKENS),
              'memory': config.SF.MEMORY, 'img_size': config.DATA.IMG_SIZE, 'n_images': len(labels),
              'gflops': cost, 'top1': top1}
    if len(logits) > 1:
        result['early_exit'] = early_exit_curve(logits, labels, cost)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'{config.MODEL.NAME}: {len(labels)} images  '
          + '  '.join(f'r{r + 1} {c:.3f} GF / {v:.2f}%' for r, (c, v) in enumerate(zip(cost, top1)))
          + f'  -> {a.out}')


if __name__ == '__main__':
    main()
