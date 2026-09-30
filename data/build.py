# --------------------------------------------------------
# Based on the Swin Transformer data pipeline
# (Copyright (c) 2021 Microsoft, MIT License; modified by Zhan Tong)
# --------------------------------------------------------
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import RepeatAugSampler
from timm.data.loader import PrefetchLoader
from torchvision import datasets, transforms


def heldout_indices(dataset, per_class):
    """The first `per_class` files of every class, in ImageFolder order, so the split is
    identical across runs and processes without storing a file."""
    by_class = {}
    for idx, (_, y) in enumerate(dataset.samples):
        by_class.setdefault(y, []).append(idx)
    return sorted(i for y in sorted(by_class) for i in sorted(by_class[y])[:per_class])


def build_transform(is_train, config):
    if is_train:
        # uint8 tensors; normalisation and random erasing run on the GPU (PrefetchLoader)
        return create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            use_prefetcher=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
    size = int((256 / 224) * config.DATA.IMG_SIZE)
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(config.DATA.IMG_SIZE),
        transforms.PILToTensor(),
    ])


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    root = os.path.join(config.DATA.DATA_PATH, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    num_classes = len(dataset.classes)
    n_held = config.DATA.HELDOUT_PER_CLASS
    if n_held > 0:
        if is_train:
            held = set(heldout_indices(dataset, n_held))
            dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if i not in held])
        else:
            # evaluation set = official val + the held-out train images, under the eval transform
            train = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, 'train'), transform=transform)
            dataset = torch.utils.data.ConcatDataset(
                [dataset, torch.utils.data.Subset(train, heldout_indices(train, n_held))])
    return dataset, num_classes


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config)

    world, rank = dist.get_world_size(), dist.get_rank()
    if config.DATA.REPEAT_AUG:
        sampler_train = RepeatAugSampler(dataset_train, num_replicas=world, rank=rank, shuffle=True,
                                         num_repeats=config.DATA.REPEAT_AUG_NUM)
    else:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=world, rank=rank,
                                                            shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=True)
    # every rank evaluates a disjoint strided share, so no image is counted twice
    data_loader_val = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset_val, np.arange(rank, len(dataset_val), world).tolist()),
        batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY, drop_last=False)

    device = torch.device('cuda', config.LOCAL_RANK)
    data_loader_train = PrefetchLoader(
        data_loader_train, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, device=device,
        re_prob=config.AUG.REPROB, re_mode=config.AUG.REMODE, re_count=config.AUG.RECOUNT)
    data_loader_val = PrefetchLoader(
        data_loader_val, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, device=device)

    mixup_fn = None
    if config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
