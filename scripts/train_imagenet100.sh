#!/bin/bash
# ImageNet-100 at 160 px, 2 GPUs.  Usage: bash scripts/train_imagenet100.sh /data/imagenet100 [config]
DATA=${1:?path to ImageNet-100}
CFG=${2:-configs/imagenet100/recurrent_9-16-25_memory.yaml}
torchrun --nproc_per_node=2 train.py --cfg "$CFG" --data-path "$DATA" --output output/imagenet100
