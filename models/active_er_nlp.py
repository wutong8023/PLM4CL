import torch
import torch.nn as nn
from utils.per_buffer_NLP import PERBufferNLP
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from datasets.transforms.LangevinDynamic_aug import InstanceLangevinDynamicAug
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Experience Replay for NLP.')
    # add related arguments
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    
    parser.add_argument('--auto_layer', required=False, action="store_true")
    parser.add_argument('--auto_feature', required=False, action="store_true")
    parser.add_argument('--auto_current_task', required=False, action="store_true")
    
    return parser


class ActiveErNLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'activeernlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(ActiveErNLP, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBufferNLP(self.args.buffer_size, self.device, require_label_name=False, PTM=self.args.ptm)
    
    def begin_task(self, dataset):
        pass
    
    def _observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
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
        print(buf_inputs.type())  # ????????????
        print(buf_inputs.size())  # ?????????shape???????????????
        print(buf_inputs.dim())  # ???????????????
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        pass


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        pass
    
    def forward(self, x):
        pass

