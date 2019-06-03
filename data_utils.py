import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets


# channel means and standard deviations of kaggle dataset
MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]

# for color augmentation, computed with make_pca.py
U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                  [-0.5989477, -0.02304967, -0.80036049],
                  [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)


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

    test_preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD)),
        KrizhevskyColorAugmentation(sigma=0)
    ])

    train_dataset = datasets.ImageFolder(train_path, train_preprocess)
    test_dataset = datasets.ImageFolder(test_path, test_preprocess)
    val_dataset = datasets.ImageFolder(val_path, test_preprocess)

    return train_dataset, test_dataset, val_dataset


class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.w0 = torch.as_tensor([1.36, 14.4, 6.64, 40.2, 49.6], dtype=torch.double)
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

    def __call__(self, img):
        sigma = self.sigma
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
