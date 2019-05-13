import os

import torch
import numpy as np

from config import *
from model import o_ONet
from train import train, evaluate
from data_utils import generate_data


def main():
    # network config
    CONFIG = SMALL_NET_CONFIG

    # load dataset
    train_dataset, test_dataset, val_dataset = generate_data(CONFIG['DATA_PATH'])

    save_dir = os.path.split(CONFIG['SAVE_PATH'])[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # train
    train(
        net=o_ONet,
        net_size=CONFIG['NET_SIZE'],
        feature_dim=CONFIG['FEATURE_DIM'],
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=CONFIG['EPOCHS'],
        learning_rate=CONFIG['LEARNING_RATE'],
        batch_size=CONFIG['BATCH_SIZE'],
        save_path=CONFIG['SAVE_PATH'],
        pretrained_model=CONFIG['PRETRAINED_PATH']
    )

    # test
    evaluate(CONFIG['SAVE_PATH'], test_dataset)


if __name__ == '__main__':
    main()
