# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import torch
from datasets import NAMES as DATASET_NAMES
from datasets import get_dataset
from datasets import ContinualDataset

from models import get_all_models
from models import get_model

from argparse import ArgumentParser
from utils.args import add_management_args

from utils.continual_training import train as ctrain
from utils.training import train, train_nlp
from utils.best_args import best_args
from utils.conf import set_random_seed


def main():
    parser = ArgumentParser(description='pseudoCL', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    
    args = parser.parse_known_args()[0]
    
    mod = importlib.import_module('models.' + args.model)
    
    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()
    
    if args.seed is not None:
        set_random_seed(args.seed)
    
    if args.model == 'mer': setattr(args, 'batch_size', 1)
    
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    if args.feature_layers == "top_n":
        n = args.feature_layers_n
        args.feature_layers = [i for i in range(backbone.encoder.config.num_hidden_layers)][-n:]
    elif args.feature_layers == "specified_n":
        n = args.feature_layers_n
        args.feature_layers = [[i for i in range(backbone.encoder.config.num_hidden_layers)][n]]
    else:
        args.feature_layers = [backbone.encoder.config.num_hidden_layers - 1]
        
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    
    if args.area == "NLP":
        if isinstance(dataset, ContinualDataset):
            train_nlp(model, dataset, args)
        else:
            assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
            ctrain(args)
    else:
        if isinstance(dataset, ContinualDataset):
            train(model, dataset, args)
        else:
            assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
            ctrain(args)



if __name__ == '__main__':
    main()
