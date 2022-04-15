# Copyright 2021-present, Tongtong Wu
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import random
import numpy as np
from typing import Tuple
from backbone.utils.tokenize import CustomizedTokenizer


def reservoir(num_seen_examples: int, buffer_size: int, current_label: int=None, num_seen_examples_per_class: {}=None,
                        stored_labels: torch.Tensor=None, max_limit: int = 50) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples
    
    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def reservoir_imbalance(num_seen_examples: int, buffer_size: int, current_label: int, num_seen_examples_per_class: {},
                        stored_labels: torch.Tensor, max_limit: int = 30):
    
    if num_seen_examples < buffer_size:
        return num_seen_examples
    else:
        num_curr_label_stored = torch.sum(stored_labels == current_label).item()
        # if c ≡ y_i is not a full class then
        if num_curr_label_stored < max_limit:
            # find the label with the most stored instances
            max_stored_label = torch.mode(stored_labels, 0)[0].item()
            # find the instance index with the selected label
            stored_index_with_label = torch.where(stored_labels == max_stored_label)[0]
            # select a id index
            idx = random.randint(0, stored_index_with_label.size()[0]-1)
            # return one index of the most stored labels
            return stored_index_with_label[idx].item()
        else:
            seen_examp_curr_label = num_seen_examples_per_class[current_label]
            rand_sample_ = random.random()
            if rand_sample_ <= float(num_curr_label_stored) / seen_examp_curr_label:
                stored_index_with_curr_label = torch.where(stored_labels == current_label)[0]
                idx = random.randint(0, stored_index_with_curr_label.size()[0])
                return stored_index_with_curr_label[idx].item()
            else:
                return -1
            pass
        pass
    pass


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class PERBufferNLP:
    """
    The memory buffer of rehearsal method.
    """
    
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir', tokenizer=None, require_label_name=False,
                 max_class_num=1000, proto_dim=768, gamma=0.5, require_proto=False, PTM="bert"):
        # todo: reservoir_imbalance
        assert mode in ['ring', 'reservoir', 'reservoir_imbalance']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.num_seen_examples_per_class = {}
        
        self.functional_index = eval(mode)
        self.require_label_name = require_label_name
        self.require_proto = require_proto
        self.max_class_num = max_class_num
        self.proto_dim = proto_dim
        self.gamma = gamma
        self.PTM = PTM
        
        if tokenizer is None:
            self.tokenizer = CustomizedTokenizer(self.PTM)
        else:
            self.tokenizer = tokenizer
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        elif mode == 'reservoir_imbalance':
            self.sampling_method = reservoir_imbalance
        else:
            self.sampling_method = reservoir
        
        # require label name
        if self.require_proto:
            self.init_proto()
        
        # here we remove logits and add features into memory.
        self.attributes = ['examples', 'examples_mask', 'labels', 'labels_name', 'labels_name_mask', 'features',
                           'task_labels']
    
    def init_tensors(self, examples: torch.Tensor, examples_mask: torch.Tensor, labels: torch.Tensor,
                     features: torch.Tensor, task_labels: torch.Tensor, labels_name: torch.Tensor,
                     labels_name_mask: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the token ids of label
        :param examples_mask: tensor containing the mask of token ids of label
        :param labels: tensor containing the labels
        :param labels_name: tensor containing the token ids of label
        :param features: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        # simplify the inter-action for attributes.
        for attr_str in self.attributes:
            # fetch parameter from the input
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                if not self.require_label_name and (attr_str == "labels_name" or attr_str == "labels_name_mask"):
                    continue
                # distinguish labels and features
                typ = torch.float32 if attr_str.endswith('features') else torch.int64
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))
    
    def init_proto(self):
        # arxiv initialization
        proto = torch.randn([self.max_class_num, self.proto_dim])
        std = 1.0 * math.sqrt(2.0 / (self.max_class_num + self.proto_dim))
        a = math.sqrt(3.0) * std
        self.proto = (proto * 2 - 1) * a
    
    def get_proto(self):
        # get prototype from memory
        return self.proto.to(self.device)
    
    def update_proto(self, examples, labels: torch.Tensor = None, features: torch.Tensor = None):
        """
        proto_t = (1 - gamma) * proto_{t-1} + gamma * batch_avg

        Parameters
        ----------
        examples :
        labels :
        features :

        Returns
        -------

        """
        for i in range(examples.shape[0]):
            # label types within the batch
            label_ = torch.unique(labels)
            for label in label_:
                l_num = torch.sum(labels == label)
                # feature with label
                feature_w_l = features[labels == label]
                feature_w_l = torch.reshape(feature_w_l, [-1, feature_w_l.shape[-1]])
                avg_feature_w_l = torch.sum(feature_w_l, dim=0) / l_num
                # update prototype
                self.proto[label] = (1 - self.gamma) * self.proto[label] + self.gamma * avg_feature_w_l
            pass
        pass
    
    def add_data(self, examples, examples_mask, labels=None, features=None, task_labels=None, labels_name=None,
                 labels_name_mask=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the x_idxs
        :param examples_mask: tensor containing the x_idxs_mask
        :param labels: tensor containing the labels
        :param labels_name: tensor containing the labels_name
        :param labels_name_mask: tensor containing the labels_mask
        :param features: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples=examples, examples_mask=examples_mask, labels=labels, features=features,
                              task_labels=task_labels, labels_name=labels_name, labels_name_mask=labels_name_mask)
        
        # update prototype first
        if self.require_label_name:
            self.update_proto(examples=examples, labels=labels, features=features)
        
        # update memory
        for i in range(examples.shape[0]):
            # update memory
            index = self.sampling_method(self.num_seen_examples, self.buffer_size, labels[i].item(),
                                         self.num_seen_examples_per_class, self.labels)
            self.num_seen_examples += 1
            
            l_ = labels[i].item()
            if l_ in self.num_seen_examples_per_class.keys():
                self.num_seen_examples_per_class[l_] += 1
            else:
                self.num_seen_examples_per_class[l_] = 0
                self.num_seen_examples_per_class[l_] += 1
            
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                self.examples_mask[index] = examples_mask[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if self.require_label_name:
                    if labels_name is not None:
                        self.labels_name[index] = labels_name[i].to(self.device)
                    if labels_name_mask is not None:
                        self.labels_name_mask[index] = labels_name_mask[i].to(self.device)
                if features is not None:
                    self.features[index] = features[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
    
    def get_data(self, size: int, choice=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])
        
        if choice is None:
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                      size=size, replace=False)
        
        ret_tuple = (torch.stack([self.examples[i].cpu() for i in choice]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                if not self.require_label_name and (attr_str == "labels_name" or attr_str == "labels_name_mask"):
                    continue
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        # modified: add choice
        return ret_tuple, choice
    
    
    
    
    # todo: add a new column - current feature.
    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        ret_tuple = (torch.stack([ee.cpu() for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                if not self.require_label_name and (attr_str == "labels_name" or attr_str == "labels_name_mask"):
                    continue
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple
    
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False
    
    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
        self.num_seen_examples_per_class = {}
    
    # todo: update the representation for all reserved instances. // or keep the parameter for the last state
    def update_batch_features(self, examples, labels=None, features=None, choice: np.ndarray = None):
        """
        update batch features for replayed instances
        Parameters
        ----------
        examples :
        labels :
        features :
        choice :

        Returns
        -------

        """
        assert len(choice) == examples.shape[0]
        
        # update prototype:
        if self.require_proto:
            self.update_proto(examples=examples, labels=labels, features=features)
        
        for i in range(examples.shape[0]):
            index = choice[i]  # the id of input is same with memory index.
            if index >= 0:
                if features is not None:
                    self.features[index] = features[i].to(self.device)
