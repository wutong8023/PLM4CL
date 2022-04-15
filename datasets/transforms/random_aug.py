"""


Author: Tong
Time: -0-2020
"""
import torch


class RandomAug(object):
    # add a random noise to data
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = tensor+noise
        return tensor
