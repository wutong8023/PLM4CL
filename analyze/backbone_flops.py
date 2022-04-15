"""
Runing Computation

Author: Tong
Time: 21-04-2021
"""
from backbone import supported_ptm
from argparse import ArgumentParser
from datasets import get_dataset
from backbone import get_supported_ptm
from datasets import NAMES as DATASET_NAMES
from backbone.utils.tokenize import CustomizedTokenizer
from transformers import RobertaTokenizer, RobertaModel
import torch
from ptflops import get_model_complexity_info

class Flops:
    def __init__(self, backbone, args):
        self.backbone = backbone
        self.args = args
        self.tokenizer = CustomizedTokenizer(ptm=self.args.ptm, max_len=50)
    
    def input_constructor(self, size_tuple):
        sample = "This paper frames a general prediction system as an observer traveling around a continuous space, " \
                 "measuring values at some locations, and predicting them at others. The observer is completely " \
                 "agnostic about any particular task being solved; it cares only about measurement locations and" \
                 " their values. This perspective leads to a machine learning framework in which seemingly unrelated" \
                 " tasks can be solved by a single model, by embedding their input and output variables into a" \
                 " shared space. An implementation of the framework is developed in which these variable embeddings" \
                 " are learned jointly with internal model parameters. In experiments, the approach is shown to (1)" \
                 " recover intuitive locations of variables in space and time, (2) exploit regularities across" \
                 " related datasets with completely disjoint input and output spaces, and (3) exploit regularities" \
                 " across seemingly unrelated tasks, outperforming task-specific single-task models and multi-task" \
                 " learning alternatives. The results suggest that even seemingly unrelated tasks may originate from" \
                 " similar underlying processes, a fact that the traveling observer model can use to make better" \
                 " predictions."
        if type(size_tuple) is tuple:
            size_tuple = size_tuple[-1]
        
        sample = sample.split()
        test_sample = sample[:min(len(sample), size_tuple)]
        test_sample = " ".join(test_sample)
        result = self.tokenizer.customized_tokenize(test_sample)
        
        return {"x": result[0], "x_mask": result[1]}
    
    def evaluate_flops(self, size: int = 50):
        with torch.cuda.device(0):
            net = self.backbone
            macs, params = get_model_complexity_info(net, (1, size), as_strings=True,
                                                     input_constructor=self.input_constructor,
                                                     print_per_layer_stat=True, verbose=True)
            print(self.args.ptm)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    


if __name__ == '__main__':
    parser = ArgumentParser(description='flops', allow_abbrev=False)
    parser.add_argument('--dataset', default="seq-clinc150", required=False, choices=DATASET_NAMES)
    parser.add_argument('--ptm', default="bert", required=False, choices=get_supported_ptm())
    parser.add_argument('--info', default="", required=False)
    parser.add_argument('--area', default="NLP", required=False)
    parser.add_argument('--prob_l', default=-1, required=False)
    args = parser.parse_known_args()[0]
    
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    
    flop_analysis = Flops(backbone=backbone, args=args)
    flop_analysis.evaluate_flops()
