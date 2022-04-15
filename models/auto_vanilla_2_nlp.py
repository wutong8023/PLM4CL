import torch
import random
from utils.per_buffer_NLP import PERBufferNLP
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from datasets.transforms.LangevinDynamic_aug import InstanceLangevinDynamicAug
from utils.args import *
import numpy as np
import math


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Experience Replay for NLP.')
    # add related arguments
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    
    parser.add_argument('--auto_layer', required=False, action="store_true")
    parser.add_argument('--auto_feature', required=False, action="store_true")
    parser.add_argument('--auto_current_task', required=False, action="store_true")
    
    return parser


class AutoVanilla2NLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'autovanilla2nlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(AutoVanilla2NLP, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBufferNLP(self.args.buffer_size, self.device, require_label_name=False, PTM=self.args.ptm,
                                   require_proto=self.args.require_proto,
                                   mode='reservoir',
                                   max_class_num=self.net.output_size)
        self.candidate_feature_layers = [i for i in range(self.net.encoder.config.num_hidden_layers)][-6:]
        self.current_feature_layers = args.feature_layers

        self.current_task = 0
        
        self.auto_layer = args.auto_layer
        self.auto_feature = args.auto_feature
        self.auto_current_task = args.auto_current_task
        self.threshold = 0.8
    
    # def end_task(self, dataset):
    #     self.current_task += 1
    #
    def end_task(self, dataset):
        self.current_task += 1
        self.net.require_proto = True
        self.net.proto = self.buffer.proto
    
    def begin_task(self, dataset):
        self.net.require_proto = False
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask, self.current_feature_layers)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.buffer.add_data(examples=inputs,
                             examples_mask=inputs_mask,
                             labels=labels,
                             features=outputs.data,
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
