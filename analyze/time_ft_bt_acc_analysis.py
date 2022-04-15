import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import seaborn as sns

from argparse import ArgumentParser
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from utils.conf import base_path
from datasets import NAMES as DATASET_NAMES


class MainVisual:
    def __init__(self):
        self.markers = ['*-', '*:', '^-', '^:', 'o-', 'o:', 'v-', 'v:', 'x-', 'x:',
                        'o--', '*--', 'v--', '^--']
        self.ptms = ["albert", "bert", "gpt2", "roberta", "xlnet"]
        self.datasets = ["seq-clinc150", "seq-maven", "seq-webred"]
        self.settings = ["class", "task"]
        self.prob_types = ["proto", "final"]
        self.methods = ["vanilla", "ewc", "hat", "er", "derpp", "joint"]
        self.num_tasks = {"clinc150": 15, "maven": 16, "webred": 24}
        self.bsize = [200, 500, 1000]
        self.time = ["time"]
        self.ft = ["forward_transfer"]
        self.bt = ["backward_transfer"]
        self.fgt = ["forgetting"]
        
        self.selection = {
            "method": self.methods,
            "ptm": self.ptms,
            "time": self.time,
            "ft": self.ft,
            "bt": self.bt,
            "fgt": self.fgt
        }
    
    def visualize(self, xs, tags, results, x_label, y_label, out_file, title):
        for i, value in enumerate(results):
            plt.plot(xs, value, self.markers[i], label=tags[i])
        plt.legend()
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        plt.title(title.split(".")[0])
        plt.savefig(out_file)
        plt.clf()
        pass
    
    def visualize_grouped_bar(self, x_label, y_label, hue, title, data, file_path):
        sns.set_theme(style="whitegrid")
        
        # Draw a nested barplot by species and sex
        
        g = sns.catplot(
            data=data, kind="bar",
            x=x_label, y=y_label, hue=hue,
            ci="sd", palette="viridis", alpha=.6, height=6
        )
        
        sns.set(rc={"figure.dpi": 200, 'savefig.dpi': 200})
        g.despine(left=True)
        # plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=20)
        plt.xlabel(x_label, fontsize=20)
        g.set_yticklabels(size=20)
        g.set_xticklabels(size=20)
        g.set_titles(size=20)
        
        
        g.legend.set_title(hue)
        
        # plt.title(title)
        if y_label == "accuracy":
            plt.ylim(0, 105)
        
        g.savefig(file_path)
        plt.clf()
    
    def merg_data(self, datasets, setting=None, merge_all=False):
        clumns = ["PLM", "Method", "forward transfer", "backward transfer", "forgetting", "time", "dataset", "task"]
        
        all_df = None
        if not merge_all:
            for ds in datasets:
                file_name = "{dataset}_{setting}".format(dataset=ds, setting=setting)
                in_file = "./data/detail_result/{file_name}.csv".format(file_name=file_name)
                
                last_task = "task{id}".format(id=self.num_tasks[ds])
                clumns[-1] = last_task
                
                df = pandas.read_csv(in_file)
                sub_df = pandas.DataFrame(df[clumns])
                sub_df = sub_df.rename(columns={last_task: "mean accuracy"})
                
                length = len(sub_df)
                sub_df["setting"] = [setting] * length
                
                if all_df is None:
                    all_df = sub_df
                else:
                    all_df = pandas.concat([all_df, sub_df])
            return all_df
        else:
            for ds in datasets:
                for set_ in self.settings:
                    file_name = "{dataset}_{setting}".format(dataset=ds, setting=set_)
                    in_file = "./data/detail_result/{file_name}.csv".format(file_name=file_name)
                    
                    last_task = "task{id}".format(id=self.num_tasks[ds])
                    clumns[-1] = last_task
                    
                    df = pandas.read_csv(in_file)
                    sub_df = pandas.DataFrame(df[clumns])
                    sub_df = sub_df.rename(columns={last_task: "accuracy"})
                    
                    length = len(sub_df)
                    sub_df["setting"] = [set_ + "-il"] * length
                    
                    if all_df is None:
                        all_df = sub_df
                    else:
                        all_df = pandas.concat([all_df, sub_df])
            return all_df
        
    def sort_data(self, data):
        new_frame = None
        for mtd in self.methods:
            sub_frame = pandas.DataFrame(data[data["Method"] == mtd])
            
            if new_frame is None:
                new_frame = sub_frame
            else:
                new_frame = pandas.concat([new_frame, sub_frame])
        return new_frame
        
    
    def visualize_factor(self, factor, vis_by, datasets, setting):
        out_dir = os.path.join(base_path(), "figure")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, "main_{}".format(factor))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_file_name = "{fn}_{vb}_{key}_{setting}.jpg".format(fn="main_result", vb=vis_by, key=factor, setting=setting)
        out_file = os.path.join(out_dir, out_file_name)
        
        data = self.merg_data(datasets, setting)
        factor = str(factor).replace("_", " ")
        if vis_by == "ptm":
            self.visualize_grouped_bar(data=data, x_label="PLM", y_label=factor, hue="Method", file_path=out_file,
                                       title=out_file_name.split(".")[0])
        else:
            self.visualize_grouped_bar(data=data, x_label="Method", y_label=factor, hue="PLM", file_path=out_file,
                                       title=out_file_name.split(".")[0])
    
    def visual_factor_details(self):
        out_dir = os.path.join(base_path(), "figure")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, "main")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        data = self.merg_data(datasets, merge_all=True)
        data = self.sort_data(data)
        
        for mtd in self.methods:
            for ptm in self.ptms:
                out_file_name = "{ptm}_{mtd}.jpg".format(mtd=mtd, ptm=ptm)
                out_file = os.path.join(out_dir, out_file_name)
                sub_frame = pandas.DataFrame(data[data["PLM"] == ptm])
                sub_frame = pandas.DataFrame(sub_frame[sub_frame["Method"] == mtd])
                self.visualize_grouped_bar(data=sub_frame, x_label="dataset", y_label="accuracy", hue="setting",
                                           file_path=out_file,
                                           title=out_file_name.split(".")[0])
            
            out_file_name = "agg_mtd_{mtd}.jpg".format(mtd=mtd)
            out_file = os.path.join(out_dir, out_file_name)
            sub_frame = pandas.DataFrame(data[data["Method"] == mtd])
            self.visualize_grouped_bar(data=sub_frame, x_label="PLM", y_label="accuracy", hue="setting",
                                       file_path=out_file,
                                       title=out_file_name.split(".")[0])
        
        for ptm in self.ptms:
            out_file_name = "agg_ptm_{ptm}.jpg".format(ptm=ptm)
            out_file = os.path.join(out_dir, out_file_name)
            sub_frame = pandas.DataFrame(data[data["PLM"] == ptm])
            self.visualize_grouped_bar(data=sub_frame, x_label="Method", y_label="accuracy", hue="setting",
                                       file_path=out_file,
                                       title=out_file_name.split(".")[0])
        
        out_file_name = "agg_all.jpg"
        out_file = os.path.join(out_dir, out_file_name)
        self.visualize_grouped_bar(data=data, x_label="dataset", y_label="accuracy", hue="setting",
                                   file_path=out_file,
                                   title=out_file_name.split(".")[0])


if __name__ == '__main__':
    parser = ArgumentParser(description='main_visual', allow_abbrev=False)
    parser.add_argument('--info', default="", required=False)
    parser.add_argument("--vis_by", default="ptm", required=False)
    parser.add_argument("--vis_type", default="time")
    parser.add_argument("--setting", default='class', required=False)
    
    args = parser.parse_known_args()[0]
    
    setting = args.setting.split("-")[0]
    
    main_v = MainVisual()
    # if args.vis_type in visual_type[0:2]:
    #     main_v.visualize_factor(dataset_name, setting, visual_by=args.vis_by, factor=args.vis_type)
    datasets = ["clinc150", "maven", "webred"]
    
    if args.vis_type == "all":
        main_v.visual_factor_details()
    else:
        main_v.visualize_factor(args.vis_type, args.vis_by, datasets, setting)
