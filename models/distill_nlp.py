# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.per_buffer_NLP import PERBufferNLP
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from datasets.transforms.LangevinDynamic_aug import InstanceLangevinDynamicAug
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    # add related arguments
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--distill_layer', type=int, required=True,
                        help='Penalty weight.')

    parser.add_argument('--require_label_name', type=bool, required=False, default=False,
                        help='the number of pseudo instances')
    
    parser.add_argument('--auto_layer', required=False, action="store_true")
    parser.add_argument('--auto_feature', required=False, action="store_true")
    parser.add_argument('--auto_current_task', required=False, action="store_true")
    
    return parser


class DistillNLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'distillnlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(DistillNLP, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBufferNLP(self.args.buffer_size, self.device, require_label_name=False, PTM=self.args.ptm)
        
        self.feature_layers = args.feature_layers
        
        self.distill_layer = args.distill_layer
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask, self.feature_layers)
        # features = self.net.features(inputs)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        if not self.buffer.is_empty():
            # begin: Loss 3
            (m_examples, m_examples_mask, m_labels, m_features), choice1 = self.buffer.get_data(
                self.args.minibatch_size)
            # buf_outputs =
            buf_feat_gen = self.net.encoder(m_examples, m_examples_mask)
            buf_feat_gen = self.net.sentence_hidden(buf_feat_gen, [self.distill_layer])
            temp_loss = self.args.alpha * F.mse_loss(buf_feat_gen, m_features)
            
            print(f"ratio: {loss.item() / temp_loss.item()}, current task: {loss.item()}, distill: {temp_loss.item()}")
            
            loss += temp_loss
            # end: Loss 3
            
            

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        else:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        # todo:
        features = self.net.encoder(inputs, inputs_mask)
        features = self.net.sentence_hidden(features, [self.distill_layer])
        self.buffer.add_data(examples=inputs,
                             examples_mask=inputs_mask,
                             labels=labels,
                             features=features.data,
                             task_labels=task_labels,
                             labels_name=labels_name,
                             labels_name_mask=labels_mask)
        
        return loss.item()
    
    def describe(self, buf_inputs):
        """
        describe the basic information of buffer_inputs
        Parameters
        ----------
        buf_inputs : data/tensor

        Returns None
        -------
        """
        print(buf_inputs.type())  # 数据类型
        print(buf_inputs.size())  # 张量的shape，是个元组
        print(buf_inputs.dim())  # 维度的数量
