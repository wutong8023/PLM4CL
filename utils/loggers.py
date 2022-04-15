# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np
import pandas as pd
import glob

useless_args = ['tensorboard', 'validation', 'csv_log', 'notes', 'load_best_args']
reformatted_args = ['fix_layers', 'feature_layers']


def merge_csv(data_dir_name: str, out: str):
    """
    start from a data dictionary
    Parameters
    ----------
    data_dir_name :
    out :

    Returns
    -------

    """
    path = data_dir_name
    
    all_files = {}
    
    out_file_name = os.path.join(path, out + ".csv")
    # delete old file
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    
    for f_path, dirs, fs in os.walk(path):
        for f in fs:
            if f != out and f.endswith(".csv"):
                all_files[f_path.split("/")[-1]] = os.path.join(f_path, f)
    
    df = pd.DataFrame()
    
    for file_ in all_files.keys():
        file_df = pd.read_csv(all_files[file_], sep=',', parse_dates=[0], infer_datetime_format=True, header=None)
        # file_df['file_name'] = file_
        df = df.append(file_df)
    
    df.to_csv(out_file_name)


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    elif setting == 'class-il':
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Instance-IL]: {} %\n'.format(task_number, round(mean_acc_class_il, 2)),
              file=sys.stderr)


class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.time = None
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None
        
        self.prob = []
        self.prob_mask_classes = []
        self.task_prob = []
        self.task_prob_mask_classes = []
    
    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)
    
    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)
    
    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)
    
    def add_running_time(self, time):
        self.time = time
    
    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il' or self.setting == 'instance-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)
    
    def log_prob(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value for probing layer performances
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.prob.append(mean_acc)
        elif self.setting == 'domain-il' or self.setting == 'instance-il':
            mean_acc, _ = mean_acc
            self.prob.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.prob.append(mean_acc_class_il)
            self.prob_mask_classes.append(mean_acc_task_il)
    
    def log_task_prob(self, mean_acc: np.ndarray, task_id: int) -> None:
        """
        Logs a mean accuracy value for probing layer performances
        :param mean_acc: mean accuracy value
        """
        if task_id == len(self.task_prob):
            self.task_prob.append([])
            self.task_prob_mask_classes.append([])
        
        if self.setting == 'general-continual':
            self.task_prob[task_id].append(mean_acc)
        elif self.setting == 'domain-il' or self.setting == 'instance-il':
            mean_acc, _ = mean_acc
            self.task_prob[task_id].append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.task_prob[task_id].append(mean_acc_class_il)
            self.task_prob_mask_classes[task_id].append(mean_acc_task_il)
    
    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]
        
        columns = list(args.keys())
        
        new_cols = []
        for i, acc in enumerate(self.accs):
            # task_id = (i+1)*args["eval_freq"]
            task_id = i + 1
            args['task' + str(task_id)] = round(acc, 2)
            new_cols.append('task' + str(i + 1))
        
        args['forward_transfer'] = round(self.fwt, 2)
        new_cols.append('forward_transfer')
        
        args['backward_transfer'] = round(self.bwt, 2)
        new_cols.append('backward_transfer')
        
        args['forgetting'] = round(self.forgetting, 2)
        new_cols.append('forgetting')
        
        args['time'] = round(self.time / 3600, 2)
        new_cols.append('time')
        
        # add prob analysis
        if args['prob_type'] != "":
            if not args["prob_all_tasks"]:
                for i, acc in enumerate(self.prob):
                    layer_id = i + 1
                    args['layer' + str(layer_id)] = round(acc, 2)
                    new_cols.append('layer' + str(i + 1))
            else:
                for i, t_prob in enumerate(self.task_prob):
                    task_id = str(i + 1)
                    for j, acc in enumerate(t_prob):
                        layer_id = str(j + 1)
                        args["t-{i}-l-{j}".format(i=task_id, j=layer_id)] = round(acc, 2)
                        new_cols.append("t-{i}-l-{j}".format(i=task_id, j=layer_id))
        
        columns = new_cols + columns
        
        create_if_not_exists(base_path() + "results/" + self.setting)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset + "/" + args["pltf"] + "-" + self.model)
        
        write_headers = False
        path = base_path() + "results/" + self.setting + "/" + self.dataset \
               + "/" + args["pltf"] + "-" + self.model + "/mean_accs.csv"
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)
        
        if self.setting == 'class-il':
            create_if_not_exists(base_path() + "results/task-il/"
                                 + self.dataset)
            create_if_not_exists(base_path() + "results/task-il/"
                                 + self.dataset + "/" + args["pltf"] + "-" + self.model)
            
            for i, acc in enumerate(self.accs_mask_classes):
                args['task' + str(i + 1)] = round(acc, 2)
            
            args['forward_transfer'] = round(self.fwt_mask_classes, 2)
            args['backward_transfer'] = round(self.bwt_mask_classes, 2)
            args['forgetting'] = round(self.forgetting_mask_classes, 2)
            
            # add prob analysis
            for i, acc in enumerate(self.prob_mask_classes):
                args['layer' + str(i + 1)] = round(acc, 2)
            
            # add task prob analysis
            for i, t_prob in enumerate(self.task_prob_mask_classes):
                task_id = str(i + 1)
                for j, acc in enumerate(t_prob):
                    layer_id = str(j + 1)
                    args["t-{i}-l-{j}".format(i=task_id, j=layer_id)] = round(acc, 2)
                
            write_headers = False
            path = base_path() + "results/task-il" + "/" + self.dataset + "/" \
                   + args["pltf"] + "-" + self.model + "/mean_accs.csv"
            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)

if __name__ == '__main__':
    # 'perm-mnist', 'seq-mnist', 'seq-cifar10', 'rot-mnist', 'seq-tinyimg', 'mnist-360', 'online-clinc150'
    for setting in ['class', 'task', 'instance']:
        for ds in ['online-clinc150', 'seq-clinc150', 'seq-maven', 'seq-webred']:
            dir_name = "./data/results{var}/{setting}-il/{ds}".format(setting=setting, ds=ds, var="")
            if os.path.exists(dir_name):
                out_name = "{setting}-il_{ds}".format(setting=setting, ds=ds)
                merge_csv(dir_name, out_name)
