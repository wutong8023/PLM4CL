"""


Author: Tong
Time: --2021
"""
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
import importlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.conf import base_path


class DatasetAnalysis:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.targets = dataset.targets
        self.lengths = dataset.distribution
        self.args = args
    
    def demonstrate(self):
        print(self.args.info)
        print("total_train_instances:", len(self.targets))
        print("total_classes:", len(np.unique(self.targets)))
        print("max_length:", np.max(self.lengths))
        
        t_count = self._process_targets()
        self._visualize(t_count, typ="t")
        l_count = self._process_distribution()
        self._visualize(l_count, typ="l")
    
    def _process_targets(self):
        unique_labels = np.unique(self.targets)
        count = []
        for label in unique_labels:
            count.append(np.sum(self.targets == label))
        count = np.array(count)
        count = np.sort(count)[::-1]
        return count
    
    def _process_distribution(self):
        count = []
        max_length = np.max(self.lengths)
        num = len(self.lengths)
        fold = max_length
        for i in range(fold):
            count.append(np.sum(self.lengths == i))
        count = np.array(count)
        return count
    
    def _visualize(self, y, typ: str):
        # replace x with your data
        file_name = self.args.info + "_" + typ + ".jpg"
        file_path = os.path.join(base_path(), "figure")
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        outfile = os.path.join(file_path, file_name)
        
        plt.figure(figsize=(100, 20), dpi=100)
        x = np.arange(len(y))
        # replace label with your data, mathing with x and y
        ax = sns.barplot(x=x, y=y, palette="Blues_d")

        plt.xticks(range(0, len(y), int(0.1*len(y))))

        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 90}
        
        if typ == "t":
            # targets
            plt.xlabel("num of classes in {dataset}".format(dataset=args.dataset.split("-")[1]), fontdict=font)
            plt.ylabel("number of instances per class", fontdict=font)
        else:
            plt.yscale('log')
            plt.xlabel("sentence length in {dataset}".format(dataset=args.dataset.split("-")[1]), fontdict=font)
            plt.ylabel("number of instances", fontdict=font)
        
        
        plt.tick_params(axis='both', which='major', labelsize=70)
        
        plt.savefig(outfile)


if __name__ == '__main__':
    parser = ArgumentParser(description='dataset', allow_abbrev=False)
    parser.add_argument('--dataset', default="seq-clinc150", required=False, choices=DATASET_NAMES)
    parser.add_argument('--info', default="", required=False)
    args = parser.parse_known_args()[0]
    
    dataset_name = args.dataset.split("-")[1]
    module = importlib.import_module('datasets.' + dataset_name)
    # here use x.lower to build such a mapping from filename to classname
    class_name = {x.lower(): x for x in module.__dir__()}[dataset_name]
    dataset_class = getattr(module, class_name)
    
    dataset_inst = dataset_class()
    da = DatasetAnalysis(dataset_inst, args)
    da.demonstrate()
