import os

import pickle
import torch
import numpy as np

from config import *
from model import o_ONet, BlendModel
from train import train_stem, train_blend, evaluate
from data_utils import generate_stem_dataset, generate_blend_dataset, create_blend_features


def main():
    # network config
    STEM_CONFIG = SMALL_NET_CONFIG
    stem(STEM_CONFIG)

    # blend step config
    # BLEND_CONFIG = BLEND_NET_CONFIG
    # blend(BLEND_CONFIG, STEM_CONFIG)


def stem(STEM_CONFIG):
    # create save path
    save_dir = os.path.split(STEM_CONFIG['SAVE_PATH'])[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load dataset
    train_dataset, test_dataset, val_dataset = generate_stem_dataset(
        STEM_CONFIG['DATA_PATH'],
        STEM_CONFIG['INPUT_SIZE'],
        STEM_CONFIG['DATA_AUGMENTATION']
    )

    # train
    model, record_epochs, accs, losses = train_stem(
        net=o_ONet,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        net_size=STEM_CONFIG['NET_SIZE'],
        input_size=STEM_CONFIG['INPUT_SIZE'],
        feature_dim=STEM_CONFIG['FEATURE_DIM'],
        epochs=STEM_CONFIG['EPOCHS'],
        learning_rate=STEM_CONFIG['LEARNING_RATE'],
        batch_size=STEM_CONFIG['BATCH_SIZE'],
        save_path=STEM_CONFIG['SAVE_PATH'],
        pretrained_model=STEM_CONFIG['PRETRAINED_PATH'],
        num_workers=STEM_CONFIG['NUM_WORKERS']
    )
    pickle.dump(
        (record_epochs, accs, losses),
        open(STEM_CONFIG['RECORD_PATH'], 'wb')
    )

    # test the stem network
    evaluate(STEM_CONFIG['SAVE_PATH'], test_dataset, STEM_CONFIG['NUM_WORKERS'])


def blend(BLEND_CONFIG, STEM_CONFIG):
    # create save path
    save_dir = os.path.split(BLEND_CONFIG['SAVE_PATH'])[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create features with different data augmentation
    create_blend_features(
        BLEND_CONFIG['MODEL_PATH'],
        BLEND_CONFIG['SOURCE_PATH'],
        BLEND_CONFIG['TARGET_PATH'],
        STEM_CONFIG['INPUT_SIZE'],
        STEM_CONFIG['DATA_AUGMENTATION'],
        BLEND_CONFIG['AUGMENTATION_TIMES']
    )
    # generate dataset
    train_dataset, test_dataset, val_dataset = generate_blend_dataset(BLEND_CONFIG['TARGET_PATH'])

    train_blend(
        BlendModel,
        train_dataset,
        val_dataset,
        BLEND_CONFIG['FEATURE_DIM'],
        BLEND_CONFIG['EPOCHS'],
        BLEND_CONFIG['LEARNING_RATE'],
        BLEND_CONFIG['BATCH_SIZE'],
        BLEND_CONFIG['SAVE_PATH']
    )

    # test the stem network
    evaluate(BLEND_CONFIG['SAVE_PATH'], test_dataset)


if __name__ == '__main__':
    main()
