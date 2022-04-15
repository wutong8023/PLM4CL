"""


Author: Tong
Time: 09-03-2020
"""

import torch
from transformers import BertModel, BertTokenizer
from backbone import import_from
from backbone import supported_ptm


class CustomizedTokenizer:
    def __init__(self, max_len=36, ptm="bert", special_token=()):
        self.ptm = ptm.lower()
        assert self.ptm in supported_ptm
        ptmTokenizer = import_from("transformers", supported_ptm[self.ptm][0] + "Tokenizer")
        self.tokenizer = ptmTokenizer.from_pretrained(supported_ptm[self.ptm][1], cache_dir="cache_model/")
        if self.ptm == "gpt2":
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # if len(special_token) > 0:
        #     self.tokenizer.add_tokens(special_token)
        self.max_len = max_len
    
    def customized_tokenize(self, sentence):
        token_dict = self.tokenizer(sentence,
                                    add_special_tokens=True,
                                    max_length=self.max_len,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt'
                                    )
        
        return token_dict['input_ids'], token_dict['attention_mask']
    
    def __len__(self):
        return len(self.tokenizer)
