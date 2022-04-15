# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.per_buffer import PERBuffer
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
    parser.add_argument('--lmd', type=float, required=True,
                        help='the weight for pseudo loss')
    parser.add_argument('--epsilon', type=float, required=False, default=0.1,
                        help='the parameter for pseudo instance sampling')
    parser.add_argument('--pseudo_size', type=int, required=True,
                        help='the number of pseudo instances')
    return parser


class Per407(ContinualModel):
    """
    feature: with only layer-wise regularization; without pseudo replay
    """
    NAME = 'per407'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(Per407, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBuffer(self.args.buffer_size, self.device)
        
        self.aug = InstanceLangevinDynamicAug(epsilon=self.args.epsilon)
    
    def observe(self, inputs, labels, not_aug_inputs):
        
        # begin: Loss 1
        outputs = self.net(inputs)
        # features = self.net.features(inputs)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        if not self.buffer.is_empty():
            # begin: Loss 2
            (buf_inputs1, buf_labels1, buf_features1), choice1 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs1)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels1)
            # end: Loss 2
            
            # begin: Loss 3
            (buf_inputs2, buf_labels2, buf_features2), choice2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_feat_gen = self.net.layer_wise_forward(buf_inputs2)
            loss += self.args.alpha * F.mse_loss(buf_feat_gen, buf_features2)
            # end: Loss 3
            
            # begin: Loss 4
            loss_temp = torch.zeros_like(loss)
            #
            (buf_inputs3, buf_labels3, _), _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_feat_gen = self.net.features(buf_inputs3)
            
            # # todo: modify the init feat
            # # temp_init_feat = buf_feat_gen
            # temp_init_feat = buf_feat_gen.detach().clone()
            # temp_label = buf_labels3
            # self.aug.update_model(self.net)
            #
            # for i in range(self.args.pseudo_size):
            #     pseudo_feature = self.aug(tensor_initial=temp_init_feat, label=temp_label)
            #     pseudo_outputs = self.net.classify(pseudo_feature)
            #     loss_temp += self.args.beta * self.loss(pseudo_outputs, buf_labels3)
            #     # update the initialization point
            #     temp_init_feat = pseudo_feature
            #
            # loss += self.args.lmd * loss_temp / self.args.pseudo_size
            # # end: Loss 4
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        else:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        features = self.net.layer_wise_forward(inputs)
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             features=features.data)
        
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
