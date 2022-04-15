# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
import numpy as np
import random
import os
from utils import create_if_not_exists
import torchvision.transforms.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt


class ValidationDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: np.ndarray,
                 transform: transforms = None, target_transform: transforms = None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            if np.max(img) < 2:
                img = Image.fromarray(np.uint8(img * 255))
            else:
                img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy())
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target



def get_train_val(train: datasets, test_transform: transforms,
                  dataset: str, val_perc: float = 0.1):
    """
    For CV
    Extract val_perc% of the training set as the validation set.
    :param train: training dataset
    :param test_transform: transformation of the test dataset
    :param dataset: dataset name
    :param val_perc: percentage of the training set to be extracted
    :return: the training set and the validation set
    """
    dataset_length = train.data.shape[0]
    directory = 'datasets/val_permutations/'
    create_if_not_exists(directory)
    file_name = dataset + '.pt'
    if os.path.exists(directory + file_name):
        perm = torch.load(directory + file_name)
    else:
        perm = torch.randperm(dataset_length)
        torch.save(perm, directory + file_name)
    train.data = train.data[perm]
    train.targets = np.array(train.targets)[perm]
    test_dataset = ValidationDataset(train.data[:int(val_perc * dataset_length)],
                                     train.targets[:int(val_perc * dataset_length)],
                                     transform=test_transform)
    train.data = train.data[int(val_perc * dataset_length):]
    train.targets = train.targets[int(val_perc * dataset_length):]
    
    return train, test_dataset


def split_data(data, label2id, train_ratio=0.667, test_ratio=0.2):
    """
    for nlp dataset
    :param data: data for split
    :param label2id: label2id
    :param train_ratio: train ratio
    :param test_ratio: test ratio
    :return: train, valid, test data
    """
    targets = np.array([label2id[instance["y"]] for instance in data])
    
    train_data = []
    valid_data = []
    test_data = []
    
    labels = np.unique(targets)
    for l_ in labels:
        l_idx = np.where(targets == l_)[0]
        l_size = len(l_idx)
        
        train_split = l_idx[:int(l_size * train_ratio)]
        test_split = l_idx[int(l_size * train_ratio): int(l_size * (train_ratio+test_ratio))]
        valid_split = l_idx[int(l_size * (train_ratio+test_ratio)):]
        
        train_data += [data[i] for i in train_split]
        valid_data += [data[i] for i in valid_split]
        test_data += [data[i] for i in test_split]
        
    return train_data, valid_data, test_data
