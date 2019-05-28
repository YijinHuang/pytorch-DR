SMALL_NET_CONFIG = {
    'NET_SIZE': 'small',
    'DATA_PATH': '../../grade/kaggle/train_data_full_crop_112',
    'SAVE_PATH': '../../grade/kaggle/pt_models/result_o_O_small.pt',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 112,
    'FEATURE_DIM': 512,
    'BATCH_SIZE': 128,
    'EPOCHS': 200
}

MEDIUM_NET_CONFIG = {
    'NET_SIZE': 'medium',
    'DATA_PATH': '../../grade/kaggle/train_data_full_crop_224',
    'SAVE_PATH': '../../grade/kaggle/pt_models/result_o_O_medium.pt',
    'PRETRAINED_PATH': '../../grade/kaggle/pt_models/result_o_O_small.pt',
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 224,
    'FEATURE_DIM': 1024,
    'BATCH_SIZE': 128,
    'EPOCHS': 200
}

LARGE_NET_CONFIG = {
    'NET_SIZE': 'large',
    'DATA_PATH': '../../grade/kaggle/train_data_full_crop_448',
    'SAVE_PATH': '../../grade/kaggle/pt_models/result_o_O_large.pt',
    'PRETRAINED_PATH': '../../grade/kaggle/pt_models/result_o_O_medium.pt',
    'LEARNING_RATE': 3e-3,
    'INPUT_SIZE': 448,
    'FEATURE_DIM': 2048,
    'BATCH_SIZE': 48,
    'EPOCHS': 250
}
