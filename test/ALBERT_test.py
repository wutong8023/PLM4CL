"""
Test of ALBERT

Author: Tong
Time: 13-04-2021
"""

from transformers import AlbertTokenizer, AlbertModel
import torch

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir="./cache_model/")
model = AlbertModel.from_pretrained('albert-base-v2', return_dict=True)
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

for i in outputs.hidden_states:
    print(i.shape)
print(len(outputs.hidden_states))