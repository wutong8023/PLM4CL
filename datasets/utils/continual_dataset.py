# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None
    NUM_INSTANCE_PER_SLICE = None
    NUM_TEST_INSTANCE_PER_SLICE = None
    NUM_VALID_INSTANCE_PER_SLICE = None
    
    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0  # seen_class_id for task-incremental learning
        self.online_i = 0  # seen slice id for instance-incremental learning, a.k.a online learning
        self.dataset = None
        
        self.args = args
        if self.args.area == "NLP":
            self.dataset = self.init_dataset() # todo: add features to CV tasks
            if self.SETTING == 'instance-il' or self.N_TASKS is None:
                self.N_TASKS = self.get_n_tasks()

    @abstractmethod
    def init_dataset(self) -> datasets:
        pass
    
    
    @abstractmethod
    def get_n_tasks(self) -> int:
        pass
    
    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass
    
    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders_for_online(train_dataset: datasets, test_dataset: datasets,
                                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    area = setting.args.area
    all_dataset = setting.dataset
    online_data = all_dataset.online_data
    sliced_online_data_idx = all_dataset.sliced_online_data_idx
    slice_size = setting.NUM_INSTANCE_PER_SLICE
    test_num_per_slice = setting.NUM_TEST_INSTANCE_PER_SLICE
    valid_num_per_slice = setting.NUM_VALID_INSTANCE_PER_SLICE
    valid = setting.args.validation
    
    assert (test_num_per_slice + valid_num_per_slice) < slice_size  # at least one instance in each slice
    
    slice_num_per_task = max(1, int(setting.args.batch_size / (slice_size - test_num_per_slice - valid_num_per_slice)))
    
    slice_idx_for_current_task = setting.online_i * slice_num_per_task
    
    train_idx = []
    test_idx = []
    for slc_ in range(slice_idx_for_current_task,
                      min(slice_idx_for_current_task + slice_num_per_task, len(sliced_online_data_idx))):
        if valid:
            train_idx.append(sliced_online_data_idx[slc_][:-2 * test_num_per_slice])
            test_idx.append(sliced_online_data_idx[slc_][-2 * test_num_per_slice:-test_num_per_slice])
        else:
            train_idx.append(sliced_online_data_idx[slc_][:-2 * test_num_per_slice])
            test_idx.append(sliced_online_data_idx[slc_][-test_num_per_slice:])
        pass
    pass
    train_idx = np.reshape(train_idx, -1)
    test_idx = np.reshape(test_idx, -1)
    
    if area == "NLP":
        train_dataset.set_data(select(online_data, train_idx, mode='id'), label2id=all_dataset.label2id)
        test_dataset.set_data(select(online_data, test_idx, mode='id'), label2id=all_dataset.label2id)
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.batch_size, shuffle=True, num_workers=1,
                                  collate_fn=train_dataset.collate_fn)
        test_loader = DataLoader(test_dataset,
                                 batch_size=setting.args.batch_size, shuffle=False, num_workers=1,
                                 collate_fn=train_dataset.collate_fn)
    else:
        # todo: add online learning feature for CV tasks
        # todo: important!
        # train_dataset.data = train_dataset.data[train_mask]
        # test_dataset.data = test_dataset.data[test_mask]
        #
        # train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        # test_dataset.targets = np.array(test_dataset.targets)[test_mask]
        #
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.batch_size, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_dataset,
                                 batch_size=setting.args.batch_size, shuffle=False, num_workers=1)
    
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    
    setting.online_i += 1
    return train_loader, test_loader


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks for CV & NLP.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    area = setting.args.area
    
    if area == "NLP":
        train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                    np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
        test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                                   np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
        
        train_dataset.set_data(select(train_dataset.data, train_mask))
        test_dataset.set_data(select(test_dataset.data, test_mask))
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.batch_size, shuffle=True, num_workers=1,
                                  collate_fn=train_dataset.collate_fn)
        test_loader = DataLoader(test_dataset,
                                 batch_size=setting.args.batch_size, shuffle=False, num_workers=1,
                                 collate_fn=train_dataset.collate_fn)
    
    else:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                    np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
        test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                                   np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
        
        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]
        
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.batch_size, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_dataset,
                                 batch_size=setting.args.batch_size, shuffle=False, num_workers=1)
    
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    
    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset, area="CV", train_mask=None) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    if train_mask is None:
        train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                    setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                    < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)
    
    if area == "NLP":
        train_dataset.set_data(select(train_dataset.data, train_mask))
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=train_dataset.collate_fn)
    else:
        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_all_previous_train_loader(train_dataset: datasets, batch_size: int,
                                  setting: ContinualDataset, area="CV", train_mask=None) -> DataLoader:
    """
    Creates a dataloader for all previous tasks.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    if train_mask is None:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= 0,
                                    np.array(train_dataset.targets) < setting.i)
    
    if area == "NLP":
        train_dataset.set_data(select(train_dataset.data, train_mask))
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=train_dataset.collate_fn)
    else:
        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def select(data_set, mask: np.ndarray, mode="mask"):
    """
    select data
    Parameters
    ----------
    data : list of datapoint
    mask : np array, 1 for selected, 0 for drop
    mode : mask or id
    Returns
    -------
    data_copy: dataset
    """
    data_copy = []
    if mode == "id":
        for i in mask:
            data_copy.append(data_set[i])
        return data_copy
    else:
        assert len(data_set) == mask.shape[0]
        for i in range(len(data_set)):
            if mask[i]:
                data_copy.append(data_set[i])
        return data_copy
