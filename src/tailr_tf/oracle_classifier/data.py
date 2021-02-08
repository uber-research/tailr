# Dataset Loading

import torch
import numpy
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
import math
import copy


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    return image.view(c, h, w)


def get_dataset(name, train=True, capacity=None):
    dataset = (TRAIN_DATASETS[name] if train else TEST_DATASETS[name])()

    if capacity is not None and len(dataset) < capacity:
        return ConcatDataset([
            copy.deepcopy(dataset) for _ in
            range(math.ceil(capacity / len(dataset)))
        ])
    else:
        return dataset


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]

TRAIN_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
}

TEST_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
}
