import torch
from utils.per_buffer_NLP import PERBufferNLP
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Experience Replay for NLP.')
    # add related arguments
    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--auto_layer', required=False, action="store_true")
    parser.add_argument('--auto_feature', required=False, action="store_true")
    parser.add_argument('--auto_current_task', required=False, action="store_true")
    
    return parser


class VanillaNLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'vanillanlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(VanillaNLP, self).__init__(backbone, loss, args, transform)
        
        self.candidate_feature_layers = [i for i in range(self.net.encoder.config.num_hidden_layers)][-6:]
        self.current_feature_layers = args.feature_layers
        self.current_task = 0

        self.auto_layer = args.auto_layer
        self.auto_feature = args.auto_feature
        self.auto_current_task = args.auto_current_task
        self.threshold = 0.8

    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask)
        # features = self.net.features(inputs)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.item()
