# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.nlp_dataset import NLPDataset
from utils.status import progress_bar

import numpy as np
import math
from torchvision import transforms
from transformers import AdamW


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--increment_joint', default=False, action='store_true',
                        help="default is conventional supervised learning, otherwise incremental joint learning")
    return parser


class JointNLP(ContinualModel):
    NAME = 'jointnlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform):
        super(JointNLP, self).__init__(backbone, loss, args, transform)
        self.seen_data = []
        self.current_task = 0
    
    def end_task(self, current_dataset):
        if current_dataset.SETTING != 'domain-il':
            self.seen_data += current_dataset.train_loader.dataset.data
            self.current_task += 1
            
            # # for non-incremental joint training
            if not self.args.increment_joint:
                if len(current_dataset.test_loaders) != current_dataset.N_TASKS:
                    return
            
            # reinit network
            self.net = current_dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = AdamW(self.net.parameters(), lr=self.args.lr)
            
            temp_dataset = NLPDataset(self.seen_data)
            loader = DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True,
                                collate_fn=temp_dataset.collate_fn)
            
            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = batch
                    
                    inputs = x_token_idxs.to(self.device)
                    inputs_mask = x_token_masks.to(self.device)
                    labels = y_idxs.to(self.device)
                    
                    self.opt.zero_grad()
                    outputs = self.net(inputs, inputs_mask)
                    loss = self.loss(outputs, labels)
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
        else:
            # todo: add domain_continual learning features to NLP tasks.
            return 0
            # self.old_data.append(dataset.train_loader)
            # # train
            # if len(dataset.test_loaders) != dataset.N_TASKS: return
            # loader_caches = [[] for _ in range(len(self.old_data))]
            # sources = torch.randint(5, (128,))
            # all_inputs = []
            # all_labels = []
            # for source in self.old_data:
            #     for x, l, _ in source:
            #         all_inputs.append(x)
            #         all_labels.append(l)
            # all_inputs = torch.cat(all_inputs)
            # all_labels = torch.cat(all_labels)
            # bs = self.args.batch_size
            # for e in range(self.args.n_epochs):
            #     order = torch.randperm(len(all_inputs))
            #     for i in range(int(math.ceil(len(all_inputs) / bs))):
            #         inputs = all_inputs[order][i * bs: (i+1) * bs]
            #         labels = all_labels[order][i * bs: (i+1) * bs]
            #         inputs, labels = inputs.to(self.device), labels.to(self.device)
            #         self.opt.zero_grad()
            #         outputs = self.net(inputs)
            #         loss = self.loss(outputs, labels.long())
            #         loss.backward()
            #         self.opt.step()
            #         progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        return 0
