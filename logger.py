# --------------------------------------------------------
# Based on Swin Transformer (Copyright (c) 2021 Microsoft, MIT License)
# --------------------------------------------------------
import functools
import logging
import os
import sys


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter('[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    if dist_rank == 0:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
