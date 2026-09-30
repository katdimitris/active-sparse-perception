#!/bin/bash
# Regenerate the figures in assets/ from the JSON files in results/ (written by tools/eval_exits.py).
python tools/plot_frontier.py --static results/imagenet100/static_n*.json \
    --recurrent results/imagenet100/recurrent_9-16-25_memory.json --labels "recurrent 9→16→25 + memory" \
    --title "ImageNet-100, 160 px (15,000 evaluation images)" --out assets/imagenet100_frontier.png
python tools/plot_frontier.py --static results/imagenet200/static_n*.json \
    --recurrent results/imagenet200/recurrent_9-16-25_memory.json --labels "recurrent 9→16→25 + memory" \
    --title "ImageNet-200, 192 px (10,000 val images)" --out assets/imagenet200_frontier.png
