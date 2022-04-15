"""


Author: Tong
Time: --2021
"""

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.PTMClassifier_HAT import PTMClassifierHAT
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='hat')
    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--lamb', type=float, required=False, default=0.75,
                        help='Penalty weight.')
    parser.add_argument('--clipgrad', type=int, required=False, default=1000,
                        help='clipgrad')
    parser.add_argument('--smax', type=int, required=False, default=400,
                        help='scale')
    parser.add_argument('--thres_cosh', type=int, required=False, default=50,
                        help='thres_cosh')
    parser.add_argument('--thres_emb', type=int, required=False, default=6,
                        help='thres_emb')
    return parser


class HATNLP(ContinualModel):
    NAME = 'hatnlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(HATNLP, self).__init__(backbone, loss, args, transform)
        self.n_task = 0
        self.n_class_per_task = 0
        self.args = args
        self.net = backbone
        self.clipgrad = self.args.clipgrad
        self.lamb = self.args.lamb
        self.smax = self.args.smax
        self.thres_cosh = self.args.thres_cosh
        self.thres_emb = self.args.thres_emb
        self.require_task_id = True
        
        self.mask_pre = None
        self.mask_back = None
        
        self.count = 0
        self.reset_gate = 100
        self.spec_mod_name = "encoder_adaptor"
        self.spec_mask_name = "hat_masks"
    
    def criterion(self, outputs, labels, mask_current):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            aux = 1 - self.mask_pre
            aux = torch.max(aux, torch.tensor(0.000001).to(self.device)) # prevent
            reg = reg + (mask_current * aux).sum()
            count = count + aux.sum()
        else:
            reg += mask_current.sum()
            count += torch.prod(torch.tensor(mask_current.size())).item()
        reg /= count
        return self.loss(outputs, labels) + self.lamb * reg, reg
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        if self.count < self.reset_gate:
            self.count = self.count + 1
        else:
            self.count = 0
        scale = (self.smax - 1 / self.smax) * self.count / self.reset_gate + 1 / self.smax

        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask, task_id=task_labels, scale=scale)
        mask_current = self.net.mask(task_labels, scale)
        loss, reg = self.criterion(outputs, labels, mask_current)
        
        self.opt.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        
        # Restrict layer gradients in backprop
        if task_labels > 0:
            # for p_name in ["weight", "bias"]:
                # para_name = spec_mod_name + "." + p_name
                # vals = self.model.get_view_for(para_name, self.mask_pre)
                # if vals is not None:
                #     self.mask_back[para_name] = 1 - vals

            for p_name in ["weight", "bias"]:
                para_name = self.spec_mod_name + "." + p_name
                if para_name in self.mask_back:
                    if para_name.endswith("weight"):
                        self.net.encoder_adaptor.weight.grad.data *= self.mask_back[para_name]
                    elif para_name.endswith("bias"):
                        self.net.encoder_adaptor.bias.grad.data *= self.mask_back[para_name]
                        
            num = torch.cosh(torch.clamp(scale * self.net.hat_masks.weight.data, -self.thres_cosh, self.thres_cosh)) + 1
            den = torch.cosh(self.net.hat_masks.weight.data) + 1
            self.net.hat_masks.weight.grad.data *= self.smax / scale * num / den
        
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        self.opt.step()
        
        self.net.hat_masks.weight.data = torch.clamp(self.net.hat_masks.weight.data, -self.thres_emb, self.thres_emb)
        
        return loss.item()
    
    def begin_task(self, dataset):
        self.count = 0
        if self.n_task == 0:
            # at the begining of continual learning, replace backbone model
            self.n_class_per_task = dataset.N_CLASSES_PER_TASK
            self.n_task = dataset.N_TASKS
            out_size = self.n_task * self.n_class_per_task
            self.net = PTMClassifierHAT(ptm=self.args.ptm, prob_l=self.args.prob_l, output_size=out_size,
                                        n_tasks=self.n_task)
            self.reset_opt()
            self.net.to(self.device)
            
    
    def end_task(self, dataset):
        current_task_id = len(dataset.test_loaders)-1
        current_task_id = torch.tensor(current_task_id, dtype=torch.int64, requires_grad=False)
        current_task_id = current_task_id.cuda()
        
        # Activations mask
        mask = self.net.mask(current_task_id, self.smax)
        mask = mask.data.clone().detach().requires_grad_(False)
        if len(dataset.test_loaders)-1 == 0:
            self.mask_pre = mask
        else:
            self.mask_pre = torch.max(self.mask_pre, mask)
        
        # Weights mask
        self.mask_back = {}

        for p_name in ["weight", "bias"]:
            para_name = self.spec_mod_name+"."+p_name
            vals = self.net.get_view_for(para_name, self.mask_pre)
            if vals is not None:
                self.mask_back[para_name] = 1 - vals
