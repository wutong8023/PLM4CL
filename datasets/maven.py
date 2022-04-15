"""


Author: Tong
Time: 18-04-2021
"""
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from utils.conf import base_path
from backbone import import_from
from backbone.utils.tokenize import CustomizedTokenizer
from backbone.PTMClassifier import PTMClassifier
from datasets.utils.validation import split_data
from datasets.utils.nlp_dataset import NLPDataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, store_masked_loaders_for_online
from datasets.utils.download_googledrive import download_file_from_google_drive as download_file


class MAVEN(NLPDataset):
    def __init__(self, download=True, data_file="maven/", mod="train", tokenizer=None, ptm='bert', slice_size: int = 15,
                 quick_load=False, filter_rate: float = 1):
        super(MAVEN, self).__init__()
        self.filter_rate = filter_rate
        # self.special_tokens = ["<t>", "</t>"]
        self.special_tokens = []
        self.ptm = ptm
        if tokenizer is None:
            self.tokenizer = CustomizedTokenizer(ptm=self.ptm, special_token=self.special_tokens, max_len=100)
        else:
            self.tokenizer = tokenizer
        
        if not quick_load:
            # MAVEN: A Massive General Domain Event Detection Dataset, EMNLP2020
            dir_path = os.path.join(base_path(), data_file)
            self.file_paths = [os.path.join(dir_path, file_name) for file_name in ["train.jsonl", "valid.jsonl"]]
            
            if download and not os.path.exists(self.file_paths[0]):
                self.maven_id = "1BNm9sqDG1vh_I2EH__nFIxjlFXx52NbV"
                download_file(self.maven_id, destination_dir=base_path(), file_name="maven")
            
            self.slice_size = slice_size
            
            # load data from files
            self.raw_data, self.label2id = self._prepare_data()

            # distribution
            self.distribution = self._stat_data(self.raw_data)
            
            # process online data
            self.online_data, self.sliced_online_data_idx = self._process_online_data(self.raw_data, self.slice_size)
            
            # process seq data
            self.train_data, self.valid_data, self.test_data = self._split_seq_data(self.raw_data, self.filter_rate)
            
            # process use mode
            if mod == "train":
                self.data = self.train_data
            elif mod == "test":
                self.data = self.test_data
            else:
                self.data = self.valid_data
            self.targets = np.array([self.label2id[item["y"]] for item in self.data])
        else:
            self.label2id = {}
            self.data = None
            self.targets = None
    
    def _prepare_data(self):
        _data = []
        for f_path in self.file_paths:
            with open(f_path, "r", encoding="utf-8") as file_in:
                for line in file_in:
                    _data.append(json.loads(line))
        pass
        
        _pair_data = []
        for i, item in enumerate(_data):
            content = item["content"]
            content_tokens_list = [c["tokens"] for c in content]
            events = item["events"]
            for j, event in enumerate(events):
                e_type = event["type"]
                mentions = event["mention"]
                for mention in mentions:
                    offset = mention["offset"]
                    sent_id = mention["sent_id"]
                    # insert token
                    target_sent = content_tokens_list[sent_id]
                    # add token
                    target_sent.insert(offset[1], "#")
                    target_sent.insert(offset[0], "#")
                    # sentence
                    sentence = " ".join(target_sent)
                    # remove token
                    target_sent.remove("#")
                    # append data point
                    _pair_data.append([sentence, e_type])
                pass
            pass
        pass
        
        # process label
        head_data, label2id = self.cut_tailed_data(_pair_data)
        head_data = self.cut_head_data(head_data, label2id)
        formatted_data = self._format_data(head_data, label2id)
        
        return formatted_data, label2id
    

class SequentialMAVEN(ContinualDataset):
    NAME = "seq-maven"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = None
    
    def get_n_tasks(self) -> int:
        if self.dataset is None:
            self.dataset = MAVEN(ptm=self.args.ptm, filter_rate=self.args.filter_rate)
        n_tasks = int(len(self.dataset.label2id) / self.N_CLASSES_PER_TASK)
        return n_tasks
    
    def get_data_loaders(self):
        # get the based dataset
        if self.dataset is None:
            self.dataset = MAVEN(ptm=self.args.ptm, filter_rate=self.args.filter_rate)
        
        # get new dataset
        train_dataset = MAVEN(quick_load=True)
        train_dataset.set_data(data=self.dataset.train_data, label2id=self.dataset.label2id)
        
        if self.args.validation:
            test_dataset = MAVEN(quick_load=True)
            test_dataset.set_data(data=self.dataset.train_data, label2id=self.dataset.label2id)
        else:
            test_dataset = MAVEN(quick_load=True)
            test_dataset.set_data(data=self.dataset.train_data, label2id=self.dataset.label2id)
        
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    def get_backbone(self):
        return PTMClassifier(output_size=len(self.dataset.label2id),
                             tokenizer=MAVEN(quick_load=True).tokenizer,
                             args=self.args)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy


class OnlineMAVEN(ContinualDataset):
    NAME = "online-maven"
    SETTING = "instance-il"
    
    NUM_INSTANCE_PER_SLICE = 5
    NUM_TEST_INSTANCE_PER_SLICE = 1
    NUM_VALID_INSTANCE_PER_SLICE = 1
    
    def get_n_tasks(self) -> int:
        n_ps = self.NUM_INSTANCE_PER_SLICE
        n_test_ps = self.NUM_TEST_INSTANCE_PER_SLICE
        n_valid_ps = self.NUM_VALID_INSTANCE_PER_SLICE
        n_train_ps = n_ps - n_test_ps - n_valid_ps
        
        if self.args.filter_rate < 1:
            n_train_ps = int(n_train_ps * self.args.filter_rate)
            n_ps = n_train_ps + n_valid_ps + n_test_ps
        
        if self.dataset is None:
            self.dataset = MAVEN(mod="train", slice_size=n_ps, ptm=self.args.ptm,
                                 filter_rate=self.args.filter_rate)
        
        slice_num_per_task = max(1, int(self.args.batch_size / n_train_ps))
        n_tasks = int(len(self.dataset.sliced_online_data_idx) / slice_num_per_task)
        return n_tasks
    
    def get_data_loaders(self):
        if self.dataset is None:
            self.dataset = MAVEN(mod="train", slice_size=self.NUM_INSTANCE_PER_SLICE, ptm=self.args.ptm)
        
        train_dataset = MAVEN(quick_load=True)
        test_dataset = MAVEN(quick_load=True)
        
        train, test = store_masked_loaders_for_online(train_dataset, test_dataset, self)
        return train, test
    
    def get_backbone(self):
        return PTMClassifier(output_size=len(self.dataset.label2id),
                             tokenizer=MAVEN(quick_load=True).tokenizer,
                             args=self.args)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
