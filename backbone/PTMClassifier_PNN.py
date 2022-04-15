# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.utils.modules import ListModule, AlphaModule
from typing import List
from transformers import BertModel, BertTokenizer
from backbone import xavier, num_flat_features, import_from, supported_ptm


class PTMClassifier_PNN(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset, equipped with lateral connection.
    """
    
    def __init__(self, input_size: int, output_size: int, old_cols: List[AlphaModule] = None, hidden_size=768,
                 ptm="bert", feature_size=768, require_proto=False, tokenizer=None, prob_l=-1) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        :param old_cols: a list of all the old columns
        """
        super(PTMClassifier_PNN, self).__init__()
        
        if old_cols is None:
            old_cols = []
        
        self.old_cols = []
        
        self.output_size = output_size
        self.hidden_size = hidden_size  # default
        self.feature_size = feature_size
        
        self.fc1 = nn.Linear(self.hidden_size, 100)
        self.classifier = nn.Linear(100, self.output_size)
        
        if len(old_cols) > 0:
            self.old_fc1s = ListModule()
            self.base_1 = nn.Sequential(
                nn.Linear(100 * len(old_cols), 100),
                nn.ReLU(),
            )
            self.base_2 = nn.Sequential(
                nn.Linear(100 * len(old_cols), 100),
                nn.ReLU(),
            )
            
            self.adaptor1 = nn.Sequential(AlphaModule(100 * len(old_cols)),
                                          self.base_1)
            self.adaptor2 = nn.Sequential(AlphaModule(100 * len(old_cols)),
                                          self.base_2)
            
            for old_col in old_cols:
                self.old_fc1s.append(
                    nn.Sequential(nn.Linear(self.hidden_size, 100), nn.ReLU()))
                self.old_fc1s[-1][0].load_state_dict(old_col.fc1.state_dict())
                
        # ptm
        self.ptm = ptm.lower()
        assert self.ptm in supported_ptm
        ptm_ = import_from("transformers", supported_ptm[self.ptm][0] + "Model")
        
        self.encoder = ptm_.from_pretrained(supported_ptm[self.ptm][1], output_hidden_states=True)
        if tokenizer is not None:
            self.encoder.resize_token_embeddings(len(tokenizer))
        self.prob_l = prob_l
        
        self.require_proto = require_proto
        
        # todo : modify net into one network
        if self.require_proto:
            # prototype-based classification
            self.net = nn.Sequential(
                self.fc1
            )
            self.net_ = nn.Sequential(
                self.encoder,
                self.fc1,
                self.encoder_adaptor,
                nn.ReLU(),
            )
        else:
            self.classifier = nn.Linear(self.feature_size, self.output_size, bias=True)
            self.net = nn.Sequential(
                self.classifier
            )
            self.net_ = nn.Sequential(
                self.encoder,
                self.classifier
            )
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        
        self.classifier.apply(xavier)
        if len(self.old_cols) > 0:
            self.adaptor1.apply(xavier)
            self.adaptor2.apply(xavier)

    def features(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_size)
        :param x_mask: mask tensor
        :return: output tensor (100)
        """
        encoding = self.encoder(x, attention_mask=x_mask,
                                output_hidden_states=True)  # [last_states; pooler_hidden; all_hidden_states]
        encoding = self._sentence_rep(encoding)  # batch_size * 768
        return encoding

    def _sentence_rep(self, encoding):
        """
        How to generate representation from ptm representation
        Args:
            encoding (the output of pretrained_language model):
        Returns:

        """
        # todo: modify into prob_layer based analysis
        # encoding = torch.mean(encoding.last_hidden_state, dim=1)
        rep_ = encoding.hidden_states[self.prob_l]
        encoding = torch.mean(rep_, dim=1)
    
        return encoding

    def classify(self, feature: torch.Tensor, proto: torch.Tensor = None) -> torch.Tensor:
        if self.require_proto:
            return feature * torch.transpose(proto, 0, 1)
        else:
            return self.classifier(feature)
        
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, proto: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        feature = self.features(x, x_mask)
        if len(self.old_cols) > 0:
            with torch.no_grad():
                fc1_kb = [old(feature) for old in self.old_fc1s]
            feature = F.relu(self.fc1(feature))
            y = self.adaptor1(torch.cat(fc1_kb, 1))
            out = self.classifier(feature, proto) + y
        else:
            feature = F.relu(self.fc1(feature))
            out = self.classifier(feature, proto)
        return out
    
    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
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
        assert new_params.size() == self.get_params().size()
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
