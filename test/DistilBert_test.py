"""
Test for DistilBert

Author: Tong
Time: 15-04-2021
"""

from transformers import DistilBertTokenizer, DistilBertModel
import torch
from ptflops import get_model_complexity_info

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir="./cache_model/")
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("Hello, my dog is cute",
                   add_special_tokens=True,
                   max_length=50,
                   padding='max_length',
                   return_attention_mask=True,
                   return_tensors='pt')
print(inputs['input_ids'])
print(inputs['attention_mask'])

outputs = model(**inputs, output_hidden_states=True)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
print("-----------------------------")
for i in outputs.hidden_states:
    print(i.shape)


def input_constructor(size_tuple=(1, 50)):
    return tokenizer("Hello, my dog is cute",
                     add_special_tokens=True,
                     max_length=50,
                     padding='max_length',
                     return_attention_mask=True,
                     return_tensors='pt')


with torch.cuda.device(0):
    net = model
    macs, params = get_model_complexity_info(net, (1, 50), as_strings=True, input_constructor=input_constructor,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
