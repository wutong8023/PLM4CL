"""
the learning rate analysis

Author: Tong
Time: 03-04-2020
"""
from typing import Any, Callable, Union

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


# todo: test
def get_all_scheduler():
    return CustomizeScheduler().types


def get_scheduler(type, optimizer):
    sche = CustomizeScheduler()
    get_sche = getattr(sche, type)
    
    return get_sche(optimizer)


class CustomizeScheduler:
    def __init__(self):
        self.types = [i for i in CustomizeScheduler.__dict__.keys() if not i.startswith("__")]
    
    def lambda_lr(self, optimizer):
        """
        Sets the learning rate of each parameter group to the initial lr times a given function.
        When last_epoch=-1, sets initial lr as lr.
        :param optimizer:
        :type optimizer:
        :param parameter:
        :type parameter:
        :return:
        :rtype:
        """
        
        def f(epoch, warm_up_num=30, warm_up_rate=1.01, gamma=0.95):
            if epoch < warm_up_num:
                return warm_up_rate ** epoch
            else:
                return gamma ** (epoch - warm_up_num)
        
        scheduler = lrs.LambdaLR(optimizer, lr_lambda=f)
        return scheduler
    
    def step_lr(self, optimizer, step_size=30, gamma=0.5):
        """
        Decays the learning rate of each parameter group by gamma every step_size epochs.
        Notice that such decay can happen simultaneously with other changes to the learning rate from outside this
        scheduler. When last_epoch=-1, sets initial lr as lr.
        :param parameter:
        :type parameter:
        :param optimizer:
        :type optimizer:
        :return:
        :rtype:
        ---------------------------------
        example:
        # Assuming optimizer uses lr = 0.05 for all groups
        # lr = 0.05     if epoch < 30
        # lr = 0.005    if 30 <= epoch < 60
        # lr = 0.0005   if 60 <= epoch < 90
        # ...
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        for epoch in range(100):
            train(...)
            validate(...)
            scheduler.step()
        """
        return lrs.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    def reduce_lr_on_plateau(self, optimizer):
        """
        Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate
        by a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement
        is seen for a ‘patience’ number of epochs, the learning rate is reduced.
        :param optimizer:
        :type optimizer:
        :return:
        :rtype:
        
        example:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(10):
            train(...)
            val_loss = validate(...)
            # Note that step should be called after validate()
            scheduler.step(val_loss)
        """
        return lrs.ReduceLROnPlateau(optimizer, 'min')
    
    # def one_cycle_lr(self, optimizer, parameter=(0.001, )):
    #     """
    #
    #     Args:
    #         optimizer ():
    #         parameter ():
    #
    #     Returns:
    #
    #     """
    #     return lrs.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(data_loader), epochs=10)
    
    def uniform(self, optimizer):
        """
        do nothing, keep the learning rate stable.
        Args:
            optimizer (torch.optim.Optimizer):
            parameter (list):

        Returns:
            scheduler
        """
        def f(epoch): return 1

        scheduler = lrs.LambdaLR(optimizer, lr_lambda=f)
        return scheduler



