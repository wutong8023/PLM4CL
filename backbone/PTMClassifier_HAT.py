# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import Iterator

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from backbone import xavier, num_flat_features, import_from, supported_ptm


class PTMClassifierHAT(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """
    def __init__(self, output_size: int, hidden_size=768, ptm="bert", feature_size=100, require_proto=False,
                 tokenizer=None, prob_l=-1, n_tasks=15, class_per_task=10) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(PTMClassifierHAT, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size  # default
        self.feature_size = feature_size
        self.n_tasks = n_tasks
        self.class_per_task = class_per_task
        
        self.ptm = ptm.lower()
        assert self.ptm in supported_ptm
        ptm_ = import_from("transformers", supported_ptm[self.ptm][0] + "Model")
        
        self.encoder = ptm_.from_pretrained(supported_ptm[self.ptm][1], output_hidden_states=True)
        # if tokenizer is not None:
        #     self.encoder.resize_token_embeddings(len(tokenizer))
        self.prob_l = prob_l
        
        self.encoder_adaptor = nn.Linear(self.hidden_size, self.feature_size)
        self.hat_masks = torch.nn.Embedding(self.n_tasks, self.feature_size)  # new
        
        self.require_proto = require_proto
        
        # todo : modify net into one network
        if self.require_proto:
            # prototype-based classification
            self.net = nn.Sequential(
                self.encoder_adaptor,
                nn.ReLU(),
            )
            self.net_ = nn.Sequential(
                self.encoder,
                self.encoder_adaptor,
                nn.ReLU(),
            )
        else:
            # self.classifier = nn.Linear(self.feature_size, self.output_size, bias=True)
            self.classifier = nn.ModuleList()
            for i in range(self.n_tasks):
                self.classifier.append(torch.nn.Linear(self.feature_size, self.class_per_task, bias=True))
            
            self.net = nn.Sequential(
                self.encoder_adaptor,
                nn.ReLU(),
                self.classifier
            )
            self.net_ = nn.Sequential(
                self.encoder,
                self.encoder_adaptor,
                nn.ReLU(),
                self.classifier
            )
        
        self.reset_parameters()
        self.gate = torch.nn.Sigmoid()
    
    # for hat
    def mask(self, task_id, scale=1):
        """
        get hat mask
        :param t:
        :param s:
        :return:
        """
        hat_mask = self.gate(scale * self.hat_masks(task_id))
        return hat_mask
    
    # for hat
    def get_view_for(self, n, masks):
        hat_masks = masks
        if n == 'encoder_adaptor.weight':
            return hat_masks.data.view(-1, 1).expand_as(self.encoder_adaptor.weight)
        elif n == 'encoder_adaptor.bias':
            return hat_masks.data.view(-1)
        return None
    
    # feature
    def features(self, x: torch.Tensor, x_mask: torch.Tensor, task_id: torch.Tensor=None, prob_l=None,
                 scale=1) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_size)
        :param x_mask: mask tensor
        :param task_id: task_id
        :param prob_l: prob_layer
        :param scale: scale
        :return: output tensor (100)
        """
        current_hat_mask = self.mask(task_id=task_id, scale=scale)
        
        encoding = self.encoder(x, attention_mask=x_mask,
                                output_hidden_states=True)  # [last_states; pooler_hidden; all_hidden_states]
        if prob_l is None:
            encoding = self._sentence_rep(encoding, self.prob_l)  # batch_size * 768
        else:
            encoding = self._sentence_rep(encoding, prob_l)  # batch_size * 768
        encoding = self.encoder_adaptor(encoding)  # batch_size *100
        
        # hat mask
        if task_id is not None:
            encoding = encoding * current_hat_mask.expand_as(encoding)
        
        encoding = torch.relu(encoding)
        return encoding
    
    def _sentence_rep(self, encoding, prob_l):
        """
        How to generate representation from ptm representation
        Args:
            encoding (the output of pretrained_language model):
        Returns:

        """
        # todo: modify into prob_layer based analysis
        # encoding = torch.mean(encoding.last_hidden_state, dim=1)
        rep_ = encoding.hidden_states[prob_l]
        encoding = torch.mean(rep_, dim=1)
        
        return encoding
    
    def classify(self, feature: torch.Tensor, proto: torch.Tensor = None) -> torch.Tensor:
        if self.require_proto:
            return feature * torch.transpose(proto, 0, 1)
        else:
            return self.classifier(feature)
    
    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)
        self.hat_masks.apply(xavier)
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, task_id:torch.Tensor=None, proto: torch.Tensor = None,
                scale=1) -> torch.Tensor:
        feature = self.features(x, x_mask, task_id=task_id, scale=scale)
        
        output_raw = []
        for clf_t in self.classifier:
            output_raw.append(clf_t(feature))
        
        output = []
        for i, o in enumerate(output_raw):
            if task_id == i:
                output.append(o)
            else:
                output.append(torch.zeros_like(o))
        
        output = torch.cat(output, dim=1)
        return output
    
    # probing
    def prob_features(self, x: torch.Tensor, x_mask: torch.Tensor, prob_l=-1) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_size)
        :param x_mask: mask tensor
        :return: output tensor (100)
        """
        encoding = self.encoder(x, attention_mask=x_mask,
                                output_hidden_states=True)  # [last_states; pooler_hidden; all_hidden_states]
        encoding = self._sentence_rep(encoding, prob_l)  # batch_size * 768
        return encoding
    
    def prob_proto_classify(self, x: torch.Tensor, x_mask: torch.Tensor, proto: torch.Tensor, prob_l: int):
        """
        classification
        :param x: batch_size * 768
        :param proto: class_size * 768
        :return: logits
        """
        encoding = self.encoder(x, attention_mask=x_mask, output_hidden_states=True)
        # [last_states; pooler_hidden; all_hidden_states]
        encoding = self._sentence_rep(encoding, prob_l)  # batch_size * 768
        proto = torch.transpose(proto, 0, 1)
        output = torch.matmul(encoding, proto)
        return output
    
    def prob_final_classify(self, x: torch.Tensor, x_mask: torch.Tensor, prob_l: int):
        feature = self.features(x, x_mask, prob_l=prob_l)
        output = self.classify(feature)  # -1 * dim_output
        return output
    
    # parameter
    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor
        """
        params = []
        for pp in list(self.net.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)
    
    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (input_size * 100
                    + 100 + 100 * 100 + 100 + 100 * output_size + output_size)
        """
        assert new_params.size() == self.net.get_params().size()
        progress = 0
        for pp in list(self.net.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params
    
    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                   + 100 * output_size + output_size)
        """
        grads = []
        for pp in list(self.net.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.net.parameters()):
            grads.append(pp.grad.view(-1))
        return grads


