"""


Author: Tong
Time: -0-2020
"""

import torch


class NormalDCAug(object):
    # add a random noise to data
    def __init__(self, base_means: torch.Tensor, base_cov: torch.Tensor):
        self.base_means = base_means
        self.base_cov = base_cov
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
    