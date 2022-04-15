"""


Author: Tong
Time: --2021
"""

from functools import partial
import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from deepspeed.profiling.flops_profiler import get_model_profile


def bert_input_constructor(input_shape, tokenizer):
    fake_seq = ""
    for _ in range(input_shape[1] - 2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * input_shape[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * input_shape[0])
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with torch.cuda.device(0):
    tokenizer = GPT2Tokenizer.from_pretrained('bert-base-uncased')
    model = GPT2ForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 128
    enable_profile = True
    if enable_profile:
        macs, params = get_model_profile(
            model,
            (batch_size, seq_len),
            input_constructor=partial(bert_input_constructor,
                                      tokenizer=tokenizer),
            print_profile=True,
            detailed=True,
        )
    else:
        inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
        outputs = model(inputs)
