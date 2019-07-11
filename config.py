SMALL_NET_CONFIG = {
    'NET_SIZE': 'small',
    'DATA_PATH': '../../dataset/train_data_full_128',
    'SAVE_PATH': '../../result/o_O_result/Res_o_O_deeper_small.pt',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 112,
    'FEATURE_DIM': 1152,
    'BATCH_SIZE': 128,
    'EPOCHS': 200,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    }
}

MEDIUM_NET_CONFIG = {
    'NET_SIZE': 'medium',
    'DATA_PATH': '../../dataset/train_data_full_256',
    'SAVE_PATH': '../../result/o_O_result/Res_o_O_deeper_medium.pt',
    'PRETRAINED_PATH': '../../result/o_O_result/Res_o_O_deeper_small.pt',
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 224,
    'FEATURE_DIM': 2304,
    'BATCH_SIZE': 128,
    'EPOCHS': 200,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    }
}

LARGE_NET_CONFIG = {
    'NET_SIZE': 'large',
    'DATA_PATH': '../../dataset/train_data_full_512',
    'SAVE_PATH': '../../result/o_O_result/Res_o_O_deeper_large.pt',
    'PRETRAINED_PATH': '../../result/o_O_result/Res_o_O_deeper_medium.pt',
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 448,
    'FEATURE_DIM': 4608,
    'BATCH_SIZE': 48,
    'EPOCHS': 250,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    }
}
