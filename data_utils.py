import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets


def generate_data(data_path):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_preprocess = transforms.Compose([
        transforms.RandomRotation((0, 360)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
