# --------------------------------------------------------
# Based on the Swin Transformer / SparseFormer configuration system
# (Copyright (c) 2021 Microsoft, MIT License; modified by Ziteng Gao & Zhan Tong)
# --------------------------------------------------------
import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# ----------------------------------------------------------------------------- data
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128              # per GPU
_C.DATA.DATA_PATH = ''                # ImageFolder root with train/ and val/
_C.DATA.IMG_SIZE = 224
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.REPEAT_AUG = True             # repeated augmentation (each image 3x per epoch, on different GPUs)
_C.DATA.REPEAT_AUG_NUM = 3
# move N images per class from train into the evaluation set (ImageNet-100 uses 100)
_C.DATA.HELDOUT_PER_CLASS = 0

# ----------------------------------------------------------------------------- model
_C.MODEL = CN()
_C.MODEL.NAME = 'sparseformer'
_C.MODEL.NUM_CLASSES = 1000           # set from the dataset
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.RESUME = ''

_C.SF = CN()
_C.SF.CONV_DIM = 96
_C.SF.TOKENS = [9, 16, 25]            # latent tokens per round; [N] = static SparseFormer
_C.SF.TOKEN_SAMPLING_POINTS = 36
_C.SF.WIDTH_CONFIGURATIONS = [192, 384]   # [focus, cortex]
_C.SF.BLOCK_SIZES = [1, 5]            # blocks per width
_C.SF.REPEATS = [4, 1]                # focus iterations per round
_C.SF.DROP_PATH_RATE = 0.2
_C.SF.HEAD_DIM = 0                    # 0: 64 heads at every width; >0: heads = width // HEAD_DIM
_C.SF.CORTEX_SAMPLES = True           # the leading cortex block also samples the image
_C.SF.MEMORY = 'none'                 # 'none' | 'cortex_ca'
_C.SF.MEMORY_STAGE = 0                # cortex block the memory read is inserted before
_C.SF.W_EARLY = 0.1                   # deep supervision: weight of the earlier exits' CE

# ----------------------------------------------------------------------------- training
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
# learning rates are for a global batch of 512 and scaled linearly with the batch
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

# ----------------------------------------------------------------------------- augmentation
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# ----------------------------------------------------------------------------- misc
_C.AMP = True
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 10
_C.PRINT_FREQ = 50
_C.SEED = 0
_C.EVAL_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print(f'=> merge config from {cfg_file}')
    config.merge_from_file(cfg_file)
    config.freeze()


def get_config(args):
    config = _C.clone()
    _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    config.TAG = args.tag or os.path.splitext(os.path.basename(args.cfg))[0]
    config.OUTPUT = os.path.join(args.output, config.TAG)
    config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    config.freeze()
    return config
