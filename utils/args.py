# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from backbone import get_supported_ptm
from utils.lr_scheduler import get_all_scheduler


# modularized arguments management
def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')

    parser.add_argument('--filter_rate', type=float, required=False, default=1,
                        choices=[0.1 * (i + 1) for i in range(10)],
                        help="float value in (0.0, 1.0), the ratio for training instances; Only support for CLINC150")
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--lr_scheduler', type=str, choices=get_all_scheduler(), required=False, default='uniform',
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--freq_eval', type=int, required=False, default=1,
                        help='evaluation frequency')
    # use gpu
    parser.add_argument('--use_gpu', type=bool, required=False, default=True, help='Define devices')
    parser.add_argument('--pltf', type=str, required=False, default="m", help='Define devices')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=100,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--area', type=str, required=True, help="CV or NLP", default="CV")
    parser.add_argument('--ptm', type=str, required=False, help="pre-trained model", default="bert",
                        choices=get_supported_ptm())
    parser.add_argument('--prob_l', type=int, required=False,
                        help="the probing layer id for analysis, default is the last layer", default=-1)
    parser.add_argument('--prob_type', required=False, default="", type=str, choices=["proto", "final", "inst"],
                        help="conduct prototype based probing.")
    parser.add_argument("--prob_all_tasks", required=False, action="store_true", help="conduct probing for all tasks")
    parser.add_argument('--info', type=str, required=True, help='introduction.')
    parser.add_argument('--eval_freq', type=int, required=False, default=1, help="how many tasks per evaluation")
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--require_proto', action="store_true", help="require_proto")
    parser.add_argument('--fix_layers', default="none", type=str, required=False, choices=("all", "none", "bottom_n"),
                        help='the layers without gradient updating.')
    parser.add_argument('--fix_layers_n', default="8", type=int, required=False,
                        help='the layers without gradient updating.')
    parser.add_argument("--feature_layers", default="last", type=str, choices=("last", "top_n", "specified_n"),
                        help="the layers used as feature layer.")
    parser.add_argument('--feature_layers_n', default="8", type=int, required=False,
                        help='the layers without gradient updating.')
    
    # # use gpu
    # parser.add_argument('--use_gpu', type=bool, required=False, default=True, help='Define devices')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')
