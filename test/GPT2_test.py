"""


Author: Tong
Time: --2021
"""
from transformers import GPT2Tokenizer, GPT2Model

import torch
from ptflops import get_model_complexity_info
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

print(inputs['input_ids'])
print(inputs['attention_mask'])

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
tokenizer.pad_token = tokenizer.eos_token

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
