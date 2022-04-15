# Copyright 2021-present, tongtong wu
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
    
    
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--buffer_per_task', type=int, required=True,
                        help='buffer size per task')
    parser.add_argument('--require_label_name', action='store_true',
                        help='the number of pseudo instances')
    parser.add_argument('--require_proto', action='store_true',
                        help='the number of pseudo instances')
    return parser


class EMARNLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'emarnlp'
    COMPATIBILITY = ['class-il', 'task-il']
    
    def __init__(self, backbone, loss, args, transform):
        super(EMARNLP, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBufferNLP(self.args.buffer_size, self.device, require_proto=self.args.require_proto,
                                   gamma=self.args.gamma, max_class_num=self.net.output_size,
                                   proto_dim=self.net.feature_size)
        
        self.aug = InstanceLangevinDynamicAug(epsilon=self.args.epsilon)
        
    def begin_task(self, dataset):
        # update on the current task
        
        
        # cluster for exemplar sampling
        
        # save data
        
        pass
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        proto = self.buffer.get_proto()
        print("proto", proto)
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask, proto)
        # features = self.net.features(inputs)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        if not self.buffer.is_empty():
            # begin: Loss 2
            
            (m_examples, m_examples_mask, m_labels, m_features), choice = self.buffer.get_data(
                self.args.minibatch_size)
            buf_outputs = self.net(m_examples, m_examples_mask, proto)
            loss += self.args.beta * self.loss(buf_outputs, m_labels)
            # end: Loss 2
        pass
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.scheduler.step()
        
        features = self.net(inputs, inputs_mask, proto)
        self.buffer.add_data(examples=inputs,
                             examples_mask=inputs_mask,
                             labels=labels,
                             features=features.data,
                             task_labels=task_labels,
                             labels_name=labels_name,
                             labels_name_mask=labels_mask
                             )
        return loss.item()
    
    def end_task(self, dataset):
        # memory consolidation
        
        pass
