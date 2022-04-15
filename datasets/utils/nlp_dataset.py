"""


Author: Tong
Time: 20-04-2021
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from torch.utils.data import DataLoader, Dataset
from utils.conf import base_path
from backbone import import_from
from backbone.utils.tokenize import CustomizedTokenizer
from backbone.PTMClassifier import PTMClassifier
from datasets.utils.validation import split_data
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, store_masked_loaders_for_online
from datasets.utils.download_googledrive import download_file_from_google_drive as download_file


class NLPDataset(Dataset):
    def __init__(self, data=None):
        self.label2id = None
        self.data = data
        self.targets = None
        self.tokenizer = None
        self.distribution = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        return instance
    
    def _prepare_data(self):
        pass
    
    def set_data(self, data, targets=None, label2id=None):
        if label2id is not None:
            self.label2id = label2id
        self.data = data
        if targets is None:
            self.targets = np.array([self.label2id[item["y"]] for item in self.data])
    
    def _filter_data(self, train_data, filter_rate):
        data = train_data
        fr = filter_rate
        targets = np.array([self.label2id[item["y"]] for item in data])
        labels = np.unique(targets)
        
        filtered_id = []
        for l_ in labels:
            index_l = np.where(targets == l_)
            idx_l = np.random.choice(index_l[0], int(len(index_l[0]) * fr))
            filtered_id.append(idx_l)
            pass
        pass
        
        filtered_id = np.reshape(filtered_id, (-1))
        filtered_data = [data[i] for i in filtered_id]
        return filtered_data
    
    def _slice_online_data(self, all_data, slice_size: int):
        all_targets = np.array([self.label2id[item["y"]] for item in all_data])
        all_labels = np.unique(all_targets)
        
        slice_data_list = []
        for l_ in all_labels:
            index = np.where(all_targets == l_)
            # -1 * slice_size
            sliced_data = np.reshape(index[0][:len(index[0]) // slice_size * slice_size], [-1, slice_size])
            slice_data_list.append(sliced_data)
        
        all_sliced_data_idx = np.reshape(slice_data_list[:len(slice_data_list) // slice_size * slice_size],
                                         (-1, slice_size))
        all_sliced_data_idx = np.random.permutation(all_sliced_data_idx)  # permutation
        
        return all_sliced_data_idx
    
    def _format_data(self, raw_data, label2id=None):
        """
        process [[x1, y1], [x2, y2], ...] into [{x1, y1, x1token, x1mask, y1token, y1mask, y_idx}, ..., ]
        Args:
            raw_data (list): list of data-label pairs
        Returns:
        processed_data: list of processed dictionary
        """
        if label2id is None:
            label2id = self.label2id
        processed_data = []
        distribution = []
        for item in raw_data:
            x = item[0]
            y = item[1]
            x_token_idx, x_token_mask = self.tokenizer.customized_tokenize(x)  # long tensor
            y_token_idx, y_token_mask = self.tokenizer.customized_tokenize(y)  # long tensor
            y_idx = torch.tensor(label2id[y], dtype=torch.int64)
            
            instance = {"x": x, "y": y,
                        "x_token_idx": x_token_idx,
                        "x_token_mask": x_token_mask,
                        "y_token_idx": y_token_idx,
                        "y_token_mask": y_token_mask,
                        "y_idx": y_idx}
            processed_data.append(instance)
            
            distribution.append(x_token_idx.shape[1])
        pass
        
        return processed_data
    
    @staticmethod
    def _stat_data(formatted_data):
        distribution = [len(item["x"].split()) for item in formatted_data]
        distribution = np.array(distribution)
        # print(np.max(distribution))
        return distribution
        
    def _process_online_data(self, raw_data, slice_size):
        # for online learning: actually the dataset is the
        online_data = raw_data
        # [-1 * slice_size]
        sliced_online_data_idx = self._slice_online_data(online_data, slice_size=slice_size)
        
        return online_data, sliced_online_data_idx
    
    def _split_seq_data(self, raw_data, filter_rate):
        # split the raw_data into three part
        train_data, valid_data, test_data = split_data(raw_data, self.label2id)
        
        if filter_rate < 1:
            train_data = self._filter_data(train_data, filter_rate)
        
        return train_data, valid_data, test_data
    
    def collate_fn(self, data):
        xs = [item["x"] for item in data]
        ys = [item["y"] for item in data]

        x_token_idxs = torch.cat([item["x_token_idx"] for item in data], dim=0)
        x_token_masks = torch.cat([item["x_token_mask"] for item in data], dim=0)
        y_token_idxs = torch.cat([item["y_token_idx"] for item in data], dim=0)
        y_token_masks = torch.cat([item["y_token_mask"] for item in data], dim=0)
        y_idxs = torch.stack([item["y_idx"] for item in data])
        
        return xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs
    
    @staticmethod
    def cut_head_data(head_pair_data, label2id,  head_size: int = 1000):
        # get remained labels
        target = []
        for i, item in enumerate(head_pair_data):
            target.append(label2id[item[1]])
        target = np.array(target)
        unique_label = np.unique(target)
        
        data = []
        for _l in unique_label:
            idx_l = np.where(target == _l)[0]
            if len(idx_l) > head_size:
                idx_l = np.random.choice(idx_l, head_size)
            data += [head_pair_data[i] for i in idx_l]
        print("cut head: ", len(data))
        return data
    
    @staticmethod
    def cut_tailed_data(pair_data, tail_size: int = 15):
        # get original label list
        label_set = set()
        for item in pair_data:
            label_set.add(item[1])
        label_list = [l_ for l_ in label_set]
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for idx, label in enumerate(label_list)}
        
        # get remained labels
        target = []
        for i, item in enumerate(pair_data):
            target.append(label2id[item[1]])
        target = np.array(target)
        unique_label = np.unique(target)
        for _l in unique_label:
            idx_l = np.where(target == _l)[0]
            if len(idx_l) < tail_size:
                id2label.pop(_l)
        
        label_list = [id2label[_l] for _l in id2label.keys()]
        
        # shuffle
        # shuffle_id = np.arange(len(label_list))
        # np.random.shuffle(shuffle_id)
        
        # label2id = {label: shuffle_id[idx] for idx, label in enumerate(label_list)}
        label2id = {label:idx for idx, label in enumerate(label_list)}
        
        # cut tail
        filtered_pair_data = [pair for pair in pair_data if pair[1] in label2id]
        return filtered_pair_data, label2id
