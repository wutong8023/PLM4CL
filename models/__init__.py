# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib


def get_all_models():
    # get model's name based on file name.
    # not only if the file is not endwith py or __init__.py
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    # here use x.lower to build such a mapping from filename to classname
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)


# get the model
def get_model(args, backbone, loss, transform=None):
    return names[args.model](backbone, loss, args, transform)
