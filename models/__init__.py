from .recurrent import RecurrentSparseFormer


def build_model(config):
    sf = config.SF
    return RecurrentSparseFormer(
        conv_dim=sf.CONV_DIM,
        token_schedule=tuple(int(v) for v in sf.TOKENS),
        token_sampling_points=sf.TOKEN_SAMPLING_POINTS,
        width_configurations=sf.WIDTH_CONFIGURATIONS,
        block_sizes=sf.BLOCK_SIZES,
        repeats=sf.REPEATS,
        drop_path_rate=sf.DROP_PATH_RATE,
        num_classes=config.MODEL.NUM_CLASSES,
        head_dim=sf.HEAD_DIM,
        cortex_samples=sf.CORTEX_SAMPLES,
        memory=sf.MEMORY,
        memory_stage=sf.MEMORY_STAGE,
    )
