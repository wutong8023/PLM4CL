# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for EWC')
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for EWC online')
    parser.add_argument('--whole_parameter', required=False, action="store_true",
                        help='whole_parameter for EWC online')
    
    parser.add_argument('--auto_layer', required=False, action="store_true")
    parser.add_argument('--auto_feature', required=False, action="store_true")
    parser.add_argument('--auto_current_task', required=False, action="store_true")
    
    return parser


class EwcOnNLP(ContinualModel):
    NAME = 'ewc_on_nlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform):
        super(EwcOnNLP, self).__init__(backbone, loss, args, transform)
        
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None
        self.net.regularize_whole()
    
    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            # self.fish = 1
            # print("max_fish:", torch.max(self.fish))
            # print("min_fish:", torch.min(self.fish))
            # print("mean_fish:", torch.mean(self.fish))
            
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty
    
    # def begin_task(self, dataset):
    #     print("ewc-44, length of test loaders: ", len(dataset.test_loaders))
    #     if len(dataset.test_loaders) < 2:
    #         self.observe = self.observe_wo_p  # warm up for continual learning without penalty
    #     else:
    #         self.observe = self.observe_w_p
    
    def end_task(self, dataset):
        
        
        fish = torch.zeros_like(self.net.get_params())
        
        for j, data in enumerate(dataset.train_loader):
            xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
            
            x_token_idxs = x_token_idxs.to(self.device)
            x_token_masks = x_token_masks.to(self.device)

            y_idxs = y_idxs.to(self.device)
            
            self.opt.zero_grad()
            output = self.net(x_token_idxs, x_token_masks)
            loss = - F.nll_loss(self.logsoft(output), y_idxs, reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
            loss = torch.mean(loss)
            loss.backward()
            fish += exp_cond_prob * self.net.get_grads() ** 2
        
        fish /= (len(dataset.train_loader) * self.args.batch_size)
        
        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += (1-self.args.gamma)*fish
        
        self.checkpoint = self.net.get_params().data.clone()
        
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        self.opt.zero_grad()
        outputs = self.net(inputs, inputs_mask)
        
        pnlt = self.penalty()
        
        loss = self.loss(outputs, labels) + self.args.e_lambda * pnlt

        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()
        
        # if self.checkpoint is not None:
        #     print(self.net.get_grads())
        #     self.opt.zero_grad()
        #     outputs = self.net(inputs, inputs_mask)
        #     loss = self.loss(outputs, labels)
        #     loss.backward()
        #     grads = self.net.get_grads()
        #
        #     print("mean_grads: ", torch.mean(grads))
        #     print("max_grads: ", torch.max(grads))
        #     print("min_grads:", torch.min(grads))
        #
        #     print(self.net.get_grads())
        #
        #     self.opt.zero_grad()
        #
        #     loss = self.penalty()
        #     loss.backward()
        #     grads = self.net.get_grads()
        #
        #     print("mean_penalty: ", torch.mean(grads))
        #     print("max_penalty: ", torch.max(grads))
        #     print("min_penalty:", torch.min(grads))
            
        return loss.item()
    