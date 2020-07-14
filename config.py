SMALL_NET_CONFIG = {
    'NET_SIZE': 'small',
    'DATA_PATH': '../../dataset/train_data_full_512',
    'SAVE_PATH': '../../result/o_O_result/o_O_small.pt',
    'RECORD_PATH': '../../result/o_O_result/o_O_small.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 112,
    'FEATURE_DIM': 512,
    'BATCH_SIZE': 128,
    'EPOCHS': 200,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
        'sigma': 0.5
    },
    "NUM_WORKERS": 32
}

MEDIUM_NET_CONFIG = {
    'NET_SIZE': 'medium',
    'DATA_PATH': '../../dataset/train_data_full_512',
    'SAVE_PATH': '../../result/o_O_result/o_O_medium.pt',
    'RECORD_PATH': '../../result/o_O_result/o_O_medium.rec',
    'PRETRAINED_PATH': '../../result/o_O_result/o_O_small.pt',
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 224,
    'FEATURE_DIM': 1024,
    'BATCH_SIZE': 128,
    'EPOCHS': 200,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 224, 40 / 224),  # 40 pixel in the report
        'sigma': 0.5
    },
    "NUM_WORKERS": 32
}

LARGE_NET_CONFIG = {
    'NET_SIZE': 'large',
    'DATA_PATH': '../../dataset/train_data_full_512',
    'SAVE_PATH': '../../result/o_O_result/o_O_large.pt',
    'RECORD_PATH': '../../result/o_O_result/o_O_large.rec',
    'PRETRAINED_PATH': '../../result/o_O_result/o_O_medium.pt',
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 448,
    'FEATURE_DIM': 2048,
    'BATCH_SIZE': 48,
    'EPOCHS': 250,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': (40 / 448, 40 / 448),  # 40 pixel in the report
        'sigma': 0.5
    },
    "NUM_WORKERS": 32
}

BLEND_NET_CONFIG = {
    'MODEL_PATH': '../../models/o_O_large.pt',
    'SOURCE_PATH': '/home/asus/Disk/yijin/ophthalmology/grade/kaggle/train_data_full_512',
    'TARGET_PATH': '/home/asus/Disk/yijin/ophthalmology/grade/kaggle/train_data_full_512_blend_feature_50',
    'AUGMENTATION_TIMES': 50,
    'SAVE_PATH': './test.pt',
    'LEARNING_RATE': 5e-4,
    'FEATURE_DIM': 4096,
    'BATCH_SIZE': 128,
    'EPOCHS': 100
}
