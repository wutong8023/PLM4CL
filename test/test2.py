#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : example.py
@Time    : 2021/3/26 上午12:07
@Author  : Luxi Xing
@Contact : xingluxixlx@gmail.com
"""
from functools import partial

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from ptflops import get_model_complexity_info
from thop import profile


def _input_constructor(input_shape, tokenizer):
    max_length = input_shape[1]
    
    # sequence for subsequent flops calculation
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    for _ in range(input_shape[0]):
        inp_seq = ""
        inputs = tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            truncation_strategy='longest_first',
        )
        print(inputs)
        
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        pad_token = tokenizer.pad_token_id
        pad_token_segment_id = 0
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        assert len(input_ids) == max_length
        model_input_ids.append(input_ids)
        model_attention_mask.append(attention_mask)
    
    labels = torch.tensor([1] * input_shape[0], dtype=torch.int64)
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {
        "input_ids": torch.tensor(model_input_ids),
        "attention_mask": torch.tensor(model_attention_mask),
    }
    inputs.update({"labels": labels})
    print([(k, v.size()) for k, v in inputs.items()])
    return inputs


def cal_plm_flops_with_ptflops(path, model_class, tok_class, batch_size, max_seq_length):
    tok = tok_class.from_pretrained(path)
    model = model_class.from_pretrained(path)
    flops_count, params_count = get_model_complexity_info(
        model,
        (batch_size, max_seq_length),
        as_strings=True,
        input_constructor=partial(_input_constructor, tokenizer=tok),
        print_per_layer_stat=False
    )
    print("%s | %s | %s" % ("[ptflops]", "Params(M)", "FLOPs(G)"))
    print("Model:  {}".format(model_class.__name__))
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops_count))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_count))


def cal_plm_flops_with_thop(path, model_class, tok_class, batch_size, max_seq_length):
    tok = tok_class.from_pretrained(path)
    model = model_class.from_pretrained(path)
    inputs = _input_constructor((batch_size, max_seq_length), tok)
    inputs_for_flops = (
        inputs.get("input_ids", None),
        inputs.get("attention_mask", None),
        inputs.get("position_ids", None),
        inputs.get("head_mask", None),
        inputs.get("input_embeds", None),
        inputs.get("labels", None),
    )
    total_ops, total_params = profile(model, inputs=inputs_for_flops, )
    print("%s | %s | %s" % ("[thop]", "Params(M)", "FLOPs(G)"))
    print("---|---|---")
    print("%s | %.2f | %.2f" % (model_class.__name__, total_params / (1000 ** 2), total_ops / (1000 ** 3)))


if __name__ == '__main__':
    PLM_PATH_BERT = "bert-large-uncased"
    PLM_PATH_ROBERTA = "roberta-large"
    PLM_PATH_GPT2 = 'gpt2'
    batch_size = 1
    max_seq_length = 128
    
    # cal_plm_flops_with_ptflops(PLM_PATH_BERT, BertForSequenceClassification, BertTokenizer, batch_size, max_seq_length)
    # cal_plm_flops_with_thop(PLM_PATH_BERT, BertForSequenceClassification, BertTokenizer, batch_size, max_seq_length)
    #
    # cal_plm_flops_with_ptflops(PLM_PATH_ROBERTA, RobertaForSequenceClassification, RobertaTokenizer, batch_size,
    #                            max_seq_length)
    # cal_plm_flops_with_thop(PLM_PATH_ROBERTA, RobertaForSequenceClassification, RobertaTokenizer, batch_size,
    #                         max_seq_length)

    cal_plm_flops_with_ptflops(PLM_PATH_GPT2, GPT2ForSequenceClassification, GPT2Tokenizer, batch_size,
                               max_seq_length)
    cal_plm_flops_with_thop(PLM_PATH_GPT2, GPT2ForSequenceClassification, GPT2Tokenizer, batch_size,
                            max_seq_length)
    
    # output FLOPs = 38.66
