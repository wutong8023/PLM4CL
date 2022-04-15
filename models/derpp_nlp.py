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
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class DerppNLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'derppnlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(DerppNLP, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBufferNLP(self.args.buffer_size, self.device, require_label_name=False, PTM=self.args.ptm)
        
        # self.aug = InstanceLangevinDynamicAug(epsilon=self.args.epsilon)
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask)
        # features = self.net.features(inputs)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        if not self.buffer.is_empty():
            # begin: Loss 2
            # todo: modify the output
            
            (m_examples, m_examples_mask, m_labels, m_features), choice1 = self.buffer.get_data(
                self.args.minibatch_size)
            buf_outputs = self.net(m_examples, m_examples_mask)
            loss += self.args.beta * self.loss(buf_outputs, m_labels)
            # end: Loss 2
            
            # begin: Loss 3
            (m_examples, m_examples_mask, m_labels, m_features), choice1 = self.buffer.get_data(
                self.args.minibatch_size)
            # buf_outputs =
            buf_feat_gen = self.net(m_examples, m_examples_mask)
            loss += self.args.alpha * F.mse_loss(buf_feat_gen, m_features)
            # end: Loss 3
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
        
        else:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
        
        self.buffer.add_data(examples=inputs,
                             examples_mask=inputs_mask,
                             labels=labels,
                             features=outputs.data,
                             task_labels=task_labels,
                             labels_name=labels_name,
                             labels_name_mask=labels_mask
                             )
        
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
