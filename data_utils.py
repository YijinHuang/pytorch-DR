import os
import math
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np


# channel means and standard deviations of kaggle dataset, computed by origin author
MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]

# for color augmentation, computed by origin author
U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                  [-0.5989477, -0.02304967, -0.80036049],
                  [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)

# set of resampling weights that yields balanced classes, computed by origin author
BALANCE_WEIGHTS = torch.tensor([1.3609453700116234, 14.378223495702006,
                                6.637566137566138, 40.235967926689575,
                                49.612994350282484], dtype=torch.double)


def generate_data(data_path, input_size, data_aug):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=data_aug['scale'],
            ratio=data_aug['stretch_ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['ratation'],
            translate=data_aug['translation_ratio'],
            scale=None,
            shear=None
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD)),
        KrizhevskyColorAugmentation(sigma=data_aug['sigma'])
    ])

    def return_func(x):
        return x

    train_dataset = datasets.ImageFolder(train_path, train_preprocess)
    test_dataset = datasets.ImageFolder(test_path, loader=return_func)
    val_dataset = datasets.ImageFolder(val_path, loader=return_func)

    return train_dataset, test_dataset, val_dataset


class EvaluationTransformer():
    def __init__(self, input_size, data_aug, aug_times):
        self.input_size = input_size
        self.data_aug = data_aug
        self.aug_times = aug_times
        self.transform_params = {
            'Crop': [],
            'Affine': [],
            'Horizontal_Flip': [],
            'Vertical_Flip': [],
            'ColorAugmentation': []
        }

    def transform(self, filepath):
        transform_params = self.transform_params
        imgs = []

        source = Image.open(filepath)
        for i in range(self.aug_times):
            img = F.resized_crop(source, *transform_params['Crop'][i], self.input_size, Image.BILINEAR)
            img = F.affine(img, *transform_params['Affine'][i], resample=False, fillcolor=0)
            if transform_params['Horizontal_Flip'][i]:
                img = F.hflip(img)
            if transform_params['Vertical_Flip'][i]:
                img = F.vflip(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(tuple(MEAN), tuple(STD))(img)
            img = KrizhevskyColorAugmentation()(img, transform_params['ColorAugmentation'][i])
            imgs.append(img)

        return torch.stack(imgs)

    def multi_transform(self, filepaths):
        imgs = []
        for filepath in filepaths:
            imgs.append(self.transform(filepath))

        return imgs

    def create_transform_params(self):
        input_size = self.input_size
        data_aug = self.data_aug
        aug_times = self.aug_times
        transform_params = self.transform_params

        for _ in range(aug_times):
            # crop
            i, j, h, w = self.create_crop_params(
                input_size,
                data_aug['scale'],
                data_aug['stretch_ratio']
            )
            transform_params['Crop'].append((i, j, h, w))

            # affine
            angle, translations, scale, shear = self.create_affine_params(
                data_aug['ratation'],
                data_aug['translation_ratio'],
                None,
                None
            )
            transform_params['Affine'].append((angle, translations, scale, shear))

            # horizontal flip
            if random.random() < 0.5:
                transform_params['Vertical_Flip'].append(True)
            else:
                transform_params['Vertical_Flip'].append(False)

            # vertical flip
            if random.random() < 0.5:
                transform_params['Horizontal_Flip'].append(True)
            else:
                transform_params['Horizontal_Flip'].append(False)

            # color augmentation
            mean = torch.tensor([0.0])
            deviation = torch.tensor([0.5])
            color_vector = torch.distributions.Normal(mean, deviation).sample((3,)).squeeze()
            transform_params['ColorAugmentation'].append(color_vector)

    def create_crop_params(self, input_size, scale, ratio):
        area = input_size[0] * input_size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= input_size[0] and h <= input_size[1]:
                i = random.randint(0, input_size[1] - h)
                j = random.randint(0, input_size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = input_size[0] / input_size[1]
        if (in_ratio < min(ratio)):
            w = input_size[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = input_size[1]
            w = h * max(ratio)
        else:  # whole image
            w = input_size[0]
            h = input_size[1]
        i = (input_size[1] - h) // 2
        j = (input_size[0] - w) // 2
        return i, j, h, w

    def create_affine_params(self, degrees, translate, scale_ranges, shears):
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * self.input_size[0]
            max_dy = translate[1] * self.input_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear


class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.w0 = BALANCE_WEIGHTS
        self.wf = torch.as_tensor([1, 2, 2, 2, 2], dtype=torch.double)
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)

    def step(self):
        self.epoch += 1
        factor = 0.975**(self.epoch - 1)
        self.weights = factor * self.w0 + (1 - factor) * self.wf
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.train_sample_weight, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        if color_vec is not None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
            color_vec = color_vec.squeeze()

        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


if __name__ == "__main__":
    from config import *
    CONFIG = LARGE_NET_CONFIG
    transformer = EvaluationTransformer(
        CONFIG['INPUT_SIZE'],
        CONFIG['DATA_AUGMENTATION'],
        20
    )
    transformer.create_transform_params()
    imgs = transformer.transform('./36_right.jpeg')
    transforms.ToPILImage()(imgs[0]).show()
