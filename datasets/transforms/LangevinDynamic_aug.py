"""


Author: Tong
Time: 09-02-2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class InstanceLangevinDynamicAug(object):
    # add a random noise to data
    def __init__(self, model: nn.Module = None, epsilon: float = 0.1):
        self.model = model
        self.epsilon = epsilon
    
    def update_model(self, model: nn.Module) -> None:
        self.model = model
    
    def __call__(self, tensor_initial: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_initial = torch.tensor(tensor_initial, requires_grad=True)
        tensor_initial.grad = None  # zero_grad
        
        predict = self.model.classify(tensor_initial)
        loss = F.cross_entropy(predict, label)
        loss.backward()
        # print(tensor_initial.grad)
        
        pseudo_instances = tensor_initial + self.epsilon / 2.0 * tensor_initial.grad
        pseudo_instances += math.sqrt(self.epsilon) * torch.randn_like(pseudo_instances)
        
        # pseudo_instances.requires_grad = False
        return pseudo_instances
