#!/bin/bash
# ImageNet-200 at 192 px, 2 GPUs.  Usage: bash scripts/train_imagenet200.sh /data/imagenet200 [config]
DATA=${1:?path to ImageNet-200}
CFG=${2:-configs/imagenet200/recurrent_9-16-25_memory.yaml}
torchrun --nproc_per_node=2 train.py --cfg "$CFG" --data-path "$DATA" --output output/imagenet200
