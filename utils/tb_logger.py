# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.conf import base_path
import os
from argparse import Namespace
from typing import Dict, Any
import numpy as np
import shutil


class TensorboardLogger:
    def __init__(self, args: Namespace, setting: str,
                 stash: Dict[Any, str] = None) -> None:
        from torch.utils.tensorboard import SummaryWriter
        
        self.settings = [setting]
        if setting == 'class-il':
            self.settings.append('task-il')
        # todo: add setting
        self.loggers = {}
        self.name = stash['model_name']
        for a_setting in self.settings:
            self.loggers[a_setting] = SummaryWriter(
                os.path.join(base_path(), 'tensorboard_runs', a_setting, self.name),
                purge_step=stash['task_idx'] * args.n_epochs + stash['epoch_idx'] + 1)
        config_text = ', '.join(
            ["%s=%s" % (name, getattr(args, name)) for name in args.__dir__()
             if not name.startswith('_')])
        for a_logger in self.loggers.values():
            a_logger.add_text('config', config_text)
    
    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name
    
    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common, mean_acc_task_il = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_task_il \
                if setting == 'task-il' else mean_acc_common
            index = 1 if setting == 'task-il' else 0
            accs = [all_accs[index][kk] for kk in range(len(all_accs[0]))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('acc_task%02d' % (kk + 1), acc,
                                    task_number * args.n_epochs)
            a_logger.add_scalar('acc_mean', mean_acc, task_number * args.n_epochs)
    
    def log_loss(self, loss: float, args: Namespace, epoch: int,
                 task_number: int, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, task_number * args.n_epochs + epoch)
    
    def log_loss_gcl(self, loss: float, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, iteration)
    
    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        for a_logger in self.loggers.values():
            a_logger.close()


if __name__ == '__main__':
    # 'perm-mnist', 'seq-mnist', 'seq-cifar10', 'rot-mnist', 'seq-tinyimg', 'mnist-360'
    for setting in ['class', 'task']:
        for ds in ['seq-cifar10', 'seq-tinyimg', 'seq-clinc150', 'seq-webred']:
            for buffer_size in [200]:

                dir_ = "./data/tensorboard_runs/{setting}-il/{ds}/"
                dir_name = dir_.format(setting=setting, ds=ds, buffer_size=buffer_size)
                out_ = "./data/tensorboard_runs/{setting}-il/{ds}/buf_{buffer_size}_all"
                out_name = out_.format(setting=setting, ds=ds, buffer_size=buffer_size)

                
                if os.path.exists(out_name):
                    # os.removedirs(remove_path)
                    for f_path, dirs, fs in os.walk(out_name):
                        for f in fs:
                            # print(os.path.join(f_path, f))
                            os.remove(os.path.join(f_path, f))
                        for dir in dirs:
                            os.removedirs(os.path.join(f_path, dir))
                # else:
                #     os.mkdir(out_name)
                #
                # files = []
                # for f_path, dirs, fs in os.walk(dir_name):
                #     for f in fs:
                #         files.append(os.path.join(f_path, f))
                #         print(os.path.join(f_path, f))
                #
                # for f in files:
                #     print(f)
                #     new_file_name = "_".join(f.split("/")[-4:])
                #     shutil.copyfile(f, os.path.join(out_name, new_file_name))
