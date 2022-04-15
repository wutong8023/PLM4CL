# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import numpy as np
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from analyze.layer_probing import prob_proto_nlp, prob_final_nlp, prob_inst_nlp
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)
            
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    
    model.net.train(status)
    return accs, accs_mask_classes


def evaluate_nlp(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    # todo: change the mask recorder
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
            
            x_token_idxs = x_token_idxs.to(model.device)
            x_token_masks = x_token_masks.to(model.device)
            y_token_idxs = y_token_idxs.to(model.device)
            y_token_masks = y_token_masks.to(model.device)
            y_idxs = y_idxs.to(model.device)
            
            task_id = torch.tensor(k, dtype=torch.int64)
            task_id = task_id.to(model.device)
            
            # todo: change the label recorder
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(x_token_idxs, x_token_masks, task_id)
            else:
                outputs = model.forward_nlp(x_token_idxs, x_token_masks, task_id)
            
            _, pred = torch.max(outputs.data, 1)
            
            correct += torch.sum(pred == y_idxs).item()
            total += y_idxs.shape[0]
            
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == y_idxs).item()
        
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    
    model.net.train(status)
    return accs, accs_mask_classes


# todo: add online learning features to CV tasks.
def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
    
    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)
    
    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t - 1] = results[t - 1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)
                
                progress_bar(i, len(train_loader), epoch, t, loss)
                
                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
                
                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0
        
        if hasattr(model, 'end_task'):
            model.end_task(dataset)
        
        #
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
    
    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)
    
    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))


def train_nlp(model: ContinualModel, dataset: ContinualDataset,
              args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
    
    dataset_copy = get_dataset(args)
    
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        # for forward transfer calculation
        random_results_class, random_results_task = evaluate_nlp(model, dataset_copy)
    
    print(file=sys.stderr)
    # start time
    start_time = time.time()
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate_nlp(model, dataset, last=True)
            results[t - 1] = results[t - 1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    # todo: to add logits
                    pass
                else:
                    xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
                    
                    x_token_idxs = x_token_idxs.to(model.device)
                    x_token_masks = x_token_masks.to(model.device)
                    y_token_idxs = y_token_idxs.to(model.device)
                    y_token_masks = y_token_masks.to(model.device)
                    y_idxs = y_idxs.to(model.device)
                    task_id = torch.tensor(t, dtype=torch.int64, requires_grad=False)
                    task_id = task_id.to(model.device)
                    
                    if model.require_task_id:
                        loss = model.observe(inputs=x_token_idxs, inputs_mask=x_token_masks, labels=y_idxs,
                                             labels_name=y_token_idxs, labels_mask=y_token_masks, task_labels=task_id)
                    else:
                        loss = model.observe(inputs=x_token_idxs, inputs_mask=x_token_masks, labels=y_idxs,
                                             labels_name=y_token_idxs, labels_mask=y_token_masks)
                
                progress_bar(i, len(train_loader), epoch, t, loss)
                
                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
                
                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0
        
        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if t % args.freq_eval == 0:
            # reduce the running freq
            # if (t+1) % args.eval_freq == 0:
            accs = evaluate_nlp(model, dataset)
            results.append(accs[0])
            results_mask_classes.append(accs[1])
    
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

            # prob_model
            if args.prob_all_tasks:
                if args.prob_type != "":
                    for prob_l in range(12):
                        prob_l += 1
                        if args.prob_type == "proto":
                            p_accs = prob_proto_nlp(model, dataset, prob_l=prob_l)
                        elif args.prob_type == "inst":
                            p_accs = prob_inst_nlp(model, dataset, prob_l=prob_l)
                        else:
                            p_accs = prob_final_nlp(model, dataset, prob_l=prob_l)
                        p_mean_acc = np.mean(p_accs, axis=1)
                        print("task {} prob_l {}: mean_acc {}, masked_mean_acc {}".format(t + 1,
                                                                                          prob_l,
                                                                                          round(p_mean_acc[0], 2),
                                                                                          round(p_mean_acc[1], 2)))
                        if args.csv_log:
                            csv_logger.log_task_prob(p_mean_acc, t)

            model_stash['mean_accs'].append(mean_acc)
            if args.csv_log:
                csv_logger.log(mean_acc)
            if args.tensorboard:
                tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
    
    running_time = time.time() - start_time
    
    # prob_model
    if args.prob_type != "" and not args.prob_all_tasks:
        for prob_l in range(12):
            prob_l += 1
            if args.prob_type == "proto":
                accs = prob_proto_nlp(model, dataset, prob_l=prob_l)
            elif args.prob_type == "inst":
                p_accs = prob_inst_nlp(model, dataset, prob_l=prob_l)
            else:
                accs = prob_final_nlp(model, dataset, prob_l=prob_l)
            mean_acc = np.mean(accs, axis=1)
            print("prob_l {}: mean_acc {}, masked_mean_acc {}".format(prob_l, mean_acc[0], mean_acc[1]))
            if args.csv_log:
                csv_logger.log_prob(mean_acc)
    
    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_running_time(running_time)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)
    
    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
