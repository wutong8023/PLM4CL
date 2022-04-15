"""


Author: Tong
Time: --2021
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os

from argparse import ArgumentParser
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from utils.conf import base_path
from datasets import NAMES as DATASET_NAMES
import seaborn as sns


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


def prob_proto_nlp(model: ContinualModel, dataset: ContinualDataset, prob_l=-1) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    with torch.no_grad():
        accs, accs_mask_classes = [], []
        
        all_features = None
        all_labels = None
        for k, test_loader in enumerate(dataset.test_loaders):
            # generate prototype
            for data in test_loader:
                xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
                
                x_token_idxs = x_token_idxs.to(model.device)
                x_token_masks = x_token_masks.to(model.device)
                y_idxs = y_idxs.to(model.device)
                
                features = model.net.prob_features(x_token_idxs, x_token_masks, prob_l)  # batch_size * 768
                
                if all_features is None:
                    all_features = features
                    all_labels = y_idxs
                else:
                    all_features = torch.cat([all_features, features], dim=0)
                    all_labels = torch.cat([all_labels, y_idxs], dim=0)
                pass
            pass
        pass
        
        # generate proto
        proto = []
        unique_l = torch.unique(all_labels)
        for l_ in range(unique_l.shape[0]):
            idx = torch.where(all_labels == l_, torch.tensor(True).to(model.device),
                              torch.tensor(False).to(model.device))
            features4l_ = all_features[idx]
            proto4l_ = torch.mean(features4l_, dim=0)
            proto.append(proto4l_)
        proto = torch.stack(proto, dim=0)
        
        all_features = None
        torch.cuda.empty_cache()
        
        # calculate accuracy
        for k, test_loader in enumerate(dataset.test_loaders):
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            
            # generate output
            for data in test_loader:
                xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
                
                x_token_idxs = x_token_idxs.to(model.device)
                x_token_masks = x_token_masks.to(model.device)
                y_idxs = y_idxs.to(model.device)
                
                outputs = model.net.prob_proto_classify(x_token_idxs, x_token_masks, proto, prob_l)
                
                _, pred = torch.max(outputs.data, dim=1)
                
                correct += torch.sum(pred == y_idxs).item()
                total += y_idxs.shape[0]
                
                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == y_idxs).item()
            
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    return accs, accs_mask_classes


def prob_inst_nlp(model: ContinualModel, dataset: ContinualDataset, prob_l=-1) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    with torch.no_grad():
        accs, accs_mask_classes = [], []
        
        all_features = None
        all_labels = None

        for k, test_loader in enumerate(dataset.test_loaders):
            # generate prototype
            for data in test_loader:
                xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
        
                x_token_idxs = x_token_idxs.to(model.device)
                x_token_masks = x_token_masks.to(model.device)
                y_idxs = y_idxs.to(model.device)
        
                features = model.net.prob_features(x_token_idxs, x_token_masks, prob_l)  # batch_size * 768
        
                if all_features is None:
                    all_features = features
                    all_labels = y_idxs
                else:
                    all_features = torch.cat([all_features, features], dim=0)
                    all_labels = torch.cat([all_labels, y_idxs], dim=0)
                pass
            pass
        pass
        
        # generate proto
        proto = []
        unique_l = torch.unique(all_labels)
        for l_ in range(unique_l.shape[0]):
            idx = torch.where(all_labels == l_, torch.tensor(True).to(model.device),
                              torch.tensor(False).to(model.device))
            
            # print(idx)

            features4l_ = all_features[idx]
            proto4l_ = features4l_[0]
            proto.append(proto4l_)
        proto = torch.stack(proto, dim=0)
        
        all_features = None
        torch.cuda.empty_cache()
        
        # calculate accuracy
        for k, test_loader in enumerate(dataset.test_loaders):
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            
            # generate output
            for data in test_loader:
                xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
                
                x_token_idxs = x_token_idxs.to(model.device)
                x_token_masks = x_token_masks.to(model.device)
                y_idxs = y_idxs.to(model.device)
                
                outputs = model.net.prob_proto_classify(x_token_idxs, x_token_masks, proto, prob_l)
                
                _, pred = torch.max(outputs.data, dim=1)
                
                correct += torch.sum(pred == y_idxs).item()
                total += y_idxs.shape[0]
                
                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == y_idxs).item()
            
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    return accs, accs_mask_classes


def prob_final_nlp(model: ContinualModel, dataset: ContinualDataset, prob_l=-1) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.net.eval()
    with torch.no_grad():
        accs, accs_mask_classes = [], []
        for k, test_loader in enumerate(dataset.test_loaders):
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            
            # generate output
            for data in test_loader:
                xs, ys, x_token_idxs, x_token_masks, y_token_idxs, y_token_masks, y_idxs = data
                
                x_token_idxs = x_token_idxs.to(model.device)
                x_token_masks = x_token_masks.to(model.device)
                y_idxs = y_idxs.to(model.device)
                
                outputs = model.net.prob_final_classify(x_token_idxs, x_token_masks, prob_l)
                
                _, pred = torch.max(outputs.data, dim=1)
                
                correct += torch.sum(pred == y_idxs).item()
                total += y_idxs.shape[0]
                
                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == y_idxs).item()
            
            accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    return accs, accs_mask_classes


class ProbVisual:
    def __init__(self):
        self.markers = ['*-', '*:', '^-', '^:', 'o-', 'o:', 'v-', 'v:', 'x-', 'x:',
                        'o--', '*--', 'v--', '^--', 'x--']
        self.ptms = ["albert", "bert", "gpt2", "roberta", "xlnet"]
        self.datasets = ["clinc150", "maven", "webred"]
        self.settings = ["class", "task", "instance"]
        self.prob_types = ["proto", "final"]
        self.methods = ["vanilla", "ewc_on", "hat", "er", "derpp", "joint"]
        self.num_tasks = {"clinc150": 15, "maven": 16, "webred": 24}
        self.bsize = [50, 100, 200, 500, 1000]
    
    def visualize(self, xs, tags, results, x_label, y_label, out_file, title, type="line"):
        if type == "line":
            for i, value in enumerate(results):
                plt.plot(xs, value, self.markers[i], label=tags[i])
            plt.legend()
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.xlabel(x_label, fontsize=15)
            plt.ylabel("accuracy", fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            
            # plt.title(title.split(".")[0])
            plt.savefig(out_file)
            plt.clf()
        else:
            x_axis = [i for i in xs]
            x_ticks = [i + 0.5 for i, x in enumerate(x_axis)]
            y_axis = [i for i in tags]
            y_ticks = [i + 0.5 for i, y in enumerate(y_axis)]
            
            sns.heatmap(results, cmap="Blues", vmin=0, vmax=100,
                        linewidth=0.3, cbar_kws={"shrink": 1})
            sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
            plt.xlabel(x_label, fontsize=20)
            plt.ylabel(y_label, fontsize=20)
            plt.xticks(x_ticks, x_axis, rotation=0, fontsize=15)
            plt.yticks(y_ticks, y_axis, rotation=0, fontsize=15)
            if y_label == "method":
                plt.yticks(y_ticks, y_axis, rotation=45, fontsize=20)
            
            
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
            plt.savefig(out_file)
            plt.clf()
        pass
    
    def prob_final_stage(self, dataset, prob_type, setting, visual_by: str = "ptm", plot_type="line"):
        file_name = "{dataset}_{prob_type}_all_{setting}".format(dataset=dataset,
                                                                 prob_type=prob_type,
                                                                 setting=setting)
        in_file = "./data/prob_results/{file_name}.csv".format(file_name=file_name)
        
        # prepare data
        df = pandas.read_csv(in_file)
        
        n_task = self.num_tasks[dataset]
        prefix_l = "t-{}-l-".format(n_task)
        prefix_t = "task{}".format(n_task)
        column_list = [prefix_l + str(i + 1) for i in range(12)]
        
        column_list.append(prefix_t)
        data = {}
        
        # select data
        if visual_by == "ptm":
            for ptm_ in self.ptms:
                tags = []
                results = []
                sub_frame = pandas.DataFrame(df[df["PLM"] == ptm_])
                for met_ in self.methods:
                    sub_sub_frame = sub_frame[sub_frame["Method"] == met_]
                    if len(sub_sub_frame) == 0:
                        continue
                    result = sub_sub_frame[column_list].values[0]
                    tags.append(met_)
                    results.append(result)
                data[ptm_] = (tags, results)
        else:
            for met_ in self.methods:
                tags = []
                results = []
                sub_frame = pandas.DataFrame(df[df["Method"] == met_])
                for ptm_ in self.ptms:
                    sub_sub_frame = sub_frame[sub_frame["PLM"] == ptm_]
                    if len(sub_sub_frame) == 0:
                        continue
                    result = sub_sub_frame[column_list].values[0]
                    tags.append(ptm_)
                    results.append(result)
                data[met_] = (tags, results)
        
        for key in data.keys():
            out_dir = os.path.join(base_path(), "figure")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dir = os.path.join(out_dir, dataset)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dir = os.path.join(out_dir, "final_stage_probing")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            
            out_file_name = "{fn}_{vb}_{key}_{setting}.jpg".format(fn="final_stage", vb=visual_by, key=key,
                                                                   setting=setting)
            out_file = os.path.join(out_dir, out_file_name)
            tags, results = data[key]
            xs = ["{l}".format(l=i + 1) for i in range(12)]
            xs.append("clf")  # final layer performance
            if plot_type == "line":
                self.visualize(xs, tags, results, "layer", "accuracy of mean clf", out_file, out_file_name,
                               type=plot_type)
            else:
                self.visualize(xs, tags, results, "layer", "method", out_file, out_file_name,
                               type=plot_type)
    
    # model - layer - task - mean/std
    def prob_layer_mean_std(self, dataset, prob_type, setting, visual_by: str = "ptm", plot_type="line"):
        file_name = "{dataset}_{prob_type}_all_{setting}".format(dataset=dataset,
                                                                 prob_type=prob_type,
                                                                 setting=setting)
        in_file = "./data/prob_results/{file_name}.csv".format(file_name=file_name)
        
        # prepare data
        df = pandas.read_csv(in_file)
        clumn_list = []
        for t in range(self.num_tasks[dataset]):
            task_id = t + 1
            for l in range(12):
                layer_id = l + 1
                clumn_list.append("t-{}-l-{}".format(task_id, layer_id))
            clumn_list.append("task{}".format(task_id))
        
        data = {}
        # select data for model
        for ptm_ in self.ptms:
            tags = []
            results = []
            sub_frame = pandas.DataFrame(df[df["PLM"] == ptm_])
            for met_ in self.methods:
                sub_sub_frame = sub_frame[sub_frame["Method"] == met_]
                if len(sub_sub_frame) == 0:
                    continue
                result = sub_sub_frame[clumn_list].values[0]
                result = result.reshape(self.num_tasks[dataset], 13)
                tags.append(met_)
                results.append(result)
            data[ptm_] = (tags, results)
        
        for key in data.keys():
            out_dir = os.path.join(base_path(), "figure")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dir = os.path.join(out_dir, dataset)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dir = os.path.join(out_dir, "overall_probing")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            
            # visualize mean
            tags, results = data[key]
            xs = [i + 1 for i in range(12)]
            xs.append("clf")
            
            mean_results = [np.mean(result, axis=0) for result in results]
            out_file_name = "{title}_{setting}_{ptm}_{type}.jpg".format(title="overall", setting=setting, ptm=key,
                                                                        type="mean")
            out_file = os.path.join(out_dir, out_file_name)
            if visual_type == "line":
                self.visualize(xs, tags, mean_results, "layer", "mean of accuracy", out_file, out_file_name,
                               type=plot_type)
            else:
                self.visualize(xs, tags, mean_results, "layer", "method", out_file, out_file_name,
                               type=plot_type)

            std_results = [np.std(result, axis=0) for result in results]
            out_file_name = "{title}_{setting}_{ptm}_{type}.jpg".format(title="overall", setting=setting, ptm=key,
                                                                        type="std")
            out_file = os.path.join(out_dir, out_file_name)
            if visual_type == "line":
                self.visualize(xs, tags, std_results, "layer", "std of accuracy", out_file, out_file_name,
                               type=plot_type)
            else:
                self.visualize(xs, tags, std_results, "layer", "method", out_file, out_file_name,
                               type=plot_type)

    def get_data(self, data: {}, in_file: str, dataset: str, tag_type: str = "mtd", mtd_list=None, mtd_bs=None):
        df = pandas.read_csv(in_file)
        clumn_list = []
        for t in range(self.num_tasks[dataset]):
            task_id = t + 1
            for l in range(12):
                layer_id = l + 1
                clumn_list.append("t-{}-l-{}".format(task_id, layer_id))
            clumn_list.append("task{}".format(task_id))
        
        # select data for model
        for ptm_ in self.ptms:
            tags = []
            results = []
            sub_frame = pandas.DataFrame(df[df["PLM"] == ptm_])
            if tag_type == "mtd":
                for mtd_ in mtd_list:
                    sub_sub_frame = sub_frame[sub_frame["Method"] == mtd_]
                    if len(sub_sub_frame) == 0:
                        continue
                    result = sub_sub_frame[clumn_list].values[0]
                    result = result.reshape(self.num_tasks[dataset], 13)
                    tags.append(mtd_)
                    results.append(result)
            elif tag_type == "bsize":
                for bs_ in self.bsize:
                    sub_sub_frame = sub_frame[sub_frame["buffer_size"] == bs_]
                    if len(sub_sub_frame) == 0:
                        continue
                    result = sub_sub_frame[clumn_list].values[0]
                    result = result.reshape(self.num_tasks[dataset], 13)
                    tags.append("{mtd}-{bs}".format(mtd=mtd_bs, bs=bs_))
                    results.append(result)
            
            if ptm_ in data.keys():
                new_tags = data[ptm_][0] + tags
                new_results = data[ptm_][1] + results
                data[ptm_] = (new_tags, new_results)
            else:
                data[ptm_] = (tags, results)
        
        return data
    
    # model - task - layer / layer - task - model
    def prob_layer_buffer_size(self, dataset, prob_type, setting, visual_by: str = "ptm", fine_grained=False,
                               mtd=None, plot_type="line"):
        data = {}
        
        if mtd is not None:
            file_name = "{dataset}_{prob_type}_all_{setting}".format(dataset=dataset,
                                                                     prob_type=prob_type,
                                                                     mtd=mtd,
                                                                     setting=setting)
            in_file = "./data/prob_results/{file_name}.csv".format(file_name=file_name)
            data = self.get_data(data, in_file=in_file, dataset=dataset, tag_type="mtd",
                                 mtd_list=["vanilla"])
            
            for m in mtd:
                file_name = "{dataset}_{prob_type}_{mtd}_buffer_size_{setting}".format(dataset=dataset,
                                                                                       prob_type=prob_type,
                                                                                       mtd=m,
                                                                                       setting=setting)
                in_file = "./data/prob_results/{file_name}.csv".format(file_name=file_name)
                
                data = self.get_data(data, in_file=in_file, dataset=dataset, tag_type="bsize", mtd_bs=m)
            
            file_name = "{dataset}_{prob_type}_all_{setting}".format(dataset=dataset,
                                                                     prob_type=prob_type,
                                                                     mtd=mtd,
                                                                     setting=setting)
            in_file = "./data/prob_results/{file_name}.csv".format(file_name=file_name)
            
            data = self.get_data(data, in_file=in_file, dataset=dataset, tag_type="mtd",
                                 mtd_list=["joint"])
        else:
            file_name = "{dataset}_{prob_type}_all_{setting}".format(dataset=dataset,
                                                                     prob_type=prob_type,
                                                                     mtd=mtd,
                                                                     setting=setting)
            in_file = "./data/prob_results/{file_name}.csv".format(file_name=file_name)
            
            data = self.get_data(data, in_file=in_file, dataset=dataset, tag_type="mtd",
                                 mtd_list=self.methods)
        
        for key in data.keys():
            if fine_grained:
                out_dir = os.path.join(base_path(), "figure")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                out_dir = os.path.join(out_dir, dataset)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if mtd is not None:
                    out_dir = os.path.join(out_dir, "bsize_{}_fine_grained_probing_vis_t".format("_".join(mtd)))
                else:
                    out_dir = os.path.join(out_dir, "all_fine_grained_probing_vis_t")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                
                # visualize mean
                tags, results = data[key]
                xs = [i + 1 for i in range(self.num_tasks[dataset])]  # 15
                tag_list = ["{}".format(i + 1) for i in range(12)]
                tag_list.append("clf")  # 13
                
                for i, mtd_name in enumerate(tags):
                    layer_results = results[i]
                    l_results = np.transpose(layer_results)  # 13 * 15
                    
                    out_file_name = "{fn}_{vb}_{key}_{setting}_vis_t.jpg".format(fn="bsize_fine", vb=mtd_name, key=key,
                                                                                 setting=setting)
                    out_file = os.path.join(out_dir, out_file_name)
                    if visual_type == "line":
                        self.visualize(xs, tag_list, l_results, "task", "accuracy", out_file, out_file_name,
                                       type=plot_type)
                    else:
                        self.visualize(xs, tag_list, l_results, "task", "layer", out_file, out_file_name,
                                       type=plot_type)

                out_dir = os.path.join(base_path(), "figure")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                out_dir = os.path.join(out_dir, dataset)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if mtd is not None:
                    out_dir = os.path.join(out_dir, "bsize_{}_fine_grained_probing_vis_l".format("_".join(mtd)))
                else:
                    out_dir = os.path.join(out_dir, "all_fine_grained_probing_vis_l")
                
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                
                tags, results = data[key]
                xs = [i + 1 for i in range(12)]  # 12
                xs.append("clf")
                tag_list = ["{}".format(i + 1) for i in range(self.num_tasks[dataset])]  # 15
                
                for i, mtd_name in enumerate(tags):
                    layer_results = results[i]
                    
                    out_file_name = "{fn}_{vb}_{key}_{setting}_vis_l.jpg".format(fn="bsize_fine", vb=mtd_name, key=key,
                                                                                 setting=setting)
                    out_file = os.path.join(out_dir, out_file_name)
                    if visual_type == "line":
                        self.visualize(xs, tag_list, layer_results, "layer", "accuracy", out_file, out_file_name,
                                       type=plot_type)
                    else:
                        self.visualize(xs, tag_list, layer_results, "layer", "task", out_file, out_file_name,
                                       type=plot_type)
            else:
                out_dir = os.path.join(base_path(), "figure")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                out_dir = os.path.join(out_dir, dataset)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if mtd is not None:
                    out_dir = os.path.join(out_dir, "bsize_{}_coarse_grained_probing_agg_t".format("_".join(mtd)))
                else:
                    out_dir = os.path.join(out_dir, "{}_coarse_grained_probing_agg_t".format("all"))
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                
                # visualize mean
                tags, results = data[key]
                xs = [i + 1 for i in range(self.num_tasks[dataset])]  # 15
                
                mean_results = [np.mean(result, axis=1) for result in results]  # 15
                out_file_name = "{fn}_{vb}_{key}_{setting}_agg_t.jpg".format(fn="bsize_coarse", vb="mean", key=key,
                                                                             setting=setting)
                out_file = os.path.join(out_dir, out_file_name)
                if visual_type == "line":
                    self.visualize(xs, tags, mean_results, "task", "mean of accuracy", out_file, out_file_name,
                                   type=plot_type)
                else:
                    self.visualize(xs, tags, mean_results, "task", "method", out_file, out_file_name,
                                   type=plot_type)
                
                std_results = [np.std(result, axis=1) for result in results]  # 15
                out_file_name = "{fn}_{vb}_{key}_{setting}_agg_t.jpg".format(fn="bsize_coarse", vb="std", key=key,
                                                                             setting=setting)
                out_file = os.path.join(out_dir, out_file_name)
                if visual_type == "line":
                    self.visualize(xs, tags, std_results, "task", "std of accuracy", out_file, out_file_name,
                                   type=plot_type)
                else:
                    self.visualize(xs, tags, std_results, "task", "method", out_file, out_file_name,
                                   type=plot_type)
                
                out_dir = os.path.join(base_path(), "figure")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                out_dir = os.path.join(out_dir, dataset)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if mtd is not None:
                    out_dir = os.path.join(out_dir, "bsize_{}_coarse_grained_probing_agg_l".format("_".join(mtd)))
                else:
                    out_dir = os.path.join(out_dir, "bsize_all_coarse_grained_probing_agg_l")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                
                # visualize mean
                tags, results = data[key]
                xs = [i + 1 for i in range(12)]  # 12
                xs.append("clf")
                
                mean_results = [np.mean(result, axis=0) for result in results]  # 13
                out_file_name = "{fn}_{vb}_{key}_{setting}_agg_l.jpg".format(fn="bsize_coarse", vb="mean", key=key,
                                                                             setting=setting)
                out_file = os.path.join(out_dir, out_file_name)
                if visual_type == "line":
                    self.visualize(xs, tags, mean_results, "layer", "mean of accuracy", out_file, out_file_name,
                                   type=plot_type)
                else:
                    self.visualize(xs, tags, mean_results, "layer", "method", out_file, out_file_name,
                                   type=plot_type)
                
                std_results = [np.std(result, axis=0) for result in results]  # 13
                out_file_name = "{fn}_{vb}_{key}_{setting}_agg_l.jpg".format(fn="bsize_coarse", vb="std", key=key,
                                                                             setting=setting)
                out_file = os.path.join(out_dir, out_file_name)
                if visual_type == "line":
                    self.visualize(xs, tags, std_results, "layer", "std of accuracy", out_file, out_file_name,
                                   type=plot_type)
                else:
                    self.visualize(xs, tags, std_results, "layer", "method", out_file, out_file_name,
                                   type=plot_type)


if __name__ == '__main__':
    visual_type = ["model_layer_mtd_final", "model_layer_mtd_mean_std",
                   "model_task_layer", "model_layer_bs_mean_std_fine",
                   "model_layer_bs_mean_std_coarse", "all_model_layer_bs_mean_std_coarse",
                   "all_model_layer_bs_mean_std"]
    
    parser = ArgumentParser(description='probing', allow_abbrev=False)
    parser.add_argument('--dataset', default="seq-clinc150", required=False, choices=DATASET_NAMES)
    parser.add_argument('--info', default="", required=False)
    parser.add_argument('--prob_type', default="proto", required=False)
    parser.add_argument("--vis_by", default="ptm", required=False)
    parser.add_argument("--prob_mtd", default="er", required=False)
    parser.add_argument("--vis_type", default="model_layer_mtd_mean", choices=visual_type)
    parser.add_argument("--setting", default='class', required=False)
    parser.add_argument("--plot_type", default='line', required=False)
    
    args = parser.parse_known_args()[0]
    
    dataset_name = args.dataset.split("-")[1]
    setting = args.setting.split("-")[0]
    
    prob_v = ProbVisual()
    
    if args.vis_type == visual_type[0]:
        prob_v.prob_final_stage(dataset=dataset_name, prob_type=args.prob_type, visual_by=args.vis_by, setting=setting,
                                plot_type=args.plot_type)
    elif args.vis_type == visual_type[1]:
        prob_v.prob_layer_mean_std(dataset=dataset_name, prob_type=args.prob_type, setting=setting,
                                   plot_type=args.plot_type)
    elif args.vis_type == visual_type[2]:
        prob_v.prob_layer_buffer_size(dataset=dataset_name, prob_type=args.prob_type, setting=setting,
                                      visual_by="ptm", fine_grained=False, plot_type=args.plot_type)
    elif args.vis_type == visual_type[3]:
        prob_v.prob_layer_buffer_size(dataset=dataset_name, prob_type=args.prob_type, setting=setting,
                                      visual_by="ptm", fine_grained=True, mtd=[args.prob_mtd], plot_type=args.plot_type)
    elif args.vis_type == visual_type[4]:
        prob_v.prob_layer_buffer_size(dataset=dataset_name, prob_type=args.prob_type, setting=setting,
                                      visual_by="ptm", fine_grained=False, mtd=[args.prob_mtd],
                                      plot_type=args.plot_type)
    elif args.vis_type == visual_type[5]:
        prob_v.prob_layer_buffer_size(dataset=dataset_name, prob_type=args.prob_type, setting=setting,
                                      visual_by="ptm", fine_grained=False, mtd=["er", "derpp"],
                                      plot_type=args.plot_type)
    elif args.vis_type == visual_type[6]:
        prob_v.prob_layer_buffer_size(dataset=dataset_name, prob_type=args.prob_type, setting=setting,
                                      visual_by="ptm", fine_grained=True, mtd=["er", "derpp"], plot_type=args.plot_type)
    else:
        print("out of options")
