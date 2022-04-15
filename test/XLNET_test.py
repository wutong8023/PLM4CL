"""
test for xlnet

Author: Tong
Time: 13-04-2021
"""

from transformers import XLNetTokenizer, XLNetModel
import torch
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', cache_dir="./cache_model/")
model = XLNetModel.from_pretrained('xlnet-base-cased')
inputs = tokenizer("Hello, my dog is cute",
                   add_special_tokens=True,
                   max_length=50,
                   padding='max_length',
                   return_attention_mask=True,
                   return_tensors='pt')
print(inputs['input_ids'])
print(inputs['attention_mask'])

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
