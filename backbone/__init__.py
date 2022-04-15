# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

# Add new pretrained language model here if you like.
# short_name: [Class_prefix, model_name_in_huggingface]
supported_ptm = {
        "bert": ["Bert", "bert-base-uncased"],
        "roberta": ["Roberta", "roberta-base"],
        "albert": ["Albert", "albert-base-v2"],
        "distilbert": ["DistilBert", "distilbert-base-uncased"],
        "xlnet": ["XLNet", "xlnet-base-cased"],
        "gpt2": ["GPT2", "gpt2"]
    }

def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to linear modules.

    :param m: the module to be initialized

    Example::
        >>> net = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        >>> net.apply(xavier)
    """
    
    if m.__class__.__name__ == 'Linear':
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def num_flat_features(x: torch.Tensor) -> int:
    """
    Computes the total number of items except the first dimension.
    :param x: input tensor
    :return: number of item from the second dimension onward
    """
    size = x.size()[1:]
    num_features = 1
    for ff in size:
        num_features *= ff
    return num_features

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

def get_supported_ptm():
    return supported_ptm.keys()
