# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from backbone import xavier, num_flat_features, import_from, supported_ptm
import numpy as np


class PTMClassifier(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """
    
    def __init__(self,
                 args,
                 output_size: int,
                 hidden_size=768,
                 feature_size=100,
                 proto=None,
                 tokenizer=None,
                 ) -> None:
        """
        
        :param output_size:
        :type output_size:
        :param feature_layers:
        :type feature_layers:
        :param hidden_size:
        :type hidden_size:
        :param ptm:
        :type ptm:
        :param feature_size:
        :type feature_size:
        :param require_proto:
        :type require_proto:
        :param prob_l:
        :type prob_l:
        :param whole_parameter:
        :type whole_parameter:
        """
        super(PTMClassifier, self).__init__()
        
        self.args = args
        self.output_size = output_size
        self.hidden_size = hidden_size  # default
        self.feature_size = feature_size
        
        self.ptm = self.args.ptm.lower()
        assert self.ptm in supported_ptm
        self.ptm_ = import_from("transformers", supported_ptm[self.ptm][0] + "Model")
        
        self.encoder = self.ptm_.from_pretrained(supported_ptm[self.ptm][1], output_hidden_states=True)
        if tokenizer is not None:
            # self.encoder.resize_token_embeddings(len(tokenizer))
            pass
        self.prob_l = self.args.prob_l
        
        self.encoder_adaptor = nn.Linear(self.hidden_size, self.feature_size)
        self.dropout = nn.Dropout(0.5)
        
        self.require_proto = self.args.require_proto
        self.require_proto = False
        self.proto = proto
        
        # prototype-based classification
        if self.require_proto:
            # prototype-based classification
            self.net = nn.Sequential(
                self.encoder_adaptor,
                self.dropout,
                nn.ReLU(),
            )
            self.net_ = nn.Sequential(
                self.encoder,
                self.encoder_adaptor,
                self.dropout,
                nn.ReLU(),
            )
        else:
            self.classifier = nn.Linear(self.feature_size, self.output_size, bias=True)
            self.net = nn.Sequential(
                self.encoder_adaptor,
                self.dropout,
                nn.ReLU(),
                self.classifier
            )
            self.net_ = nn.Sequential(
                self.encoder,
                self.encoder_adaptor,
                self.dropout,
                nn.ReLU(),
                self.classifier
            )
        
        self.head = self.net  #
        self.reset_parameters()
        
        self.feature_layers = self.args.feature_layers
        # fixme
        if self.feature_layers == "top_n":
            n = self.args.feature_layers_n
            self.feature_layers = [i for i in range(self.encoder.config.num_hidden_layers)][-n:]
        elif self.feature_layers == "specified_n":
            n = self.args.feature_layers_n
            self.feature_layers = [[i for i in range(self.encoder.config.num_hidden_layers)][n]]
        else:
            self.feature_layers = [self.encoder.config.num_hidden_layers - 1]
        
        self.fix_parameter()
    
    # regularization
    def regularize_whole(self):
        self.whole_parameter = self.args.whole_parameter
        if self.whole_parameter:
            self.net = self.net_
    
    # set prototype for classification
    def set_proto(self, proto):
        self.proto = proto
    
    # feature
    def features(self, x: torch.Tensor, x_mask: torch.Tensor, feature_layers: [] = None) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_siz
        :param x_mask: mask tensor
        :param prob_l: prob_layer
        :return: output tensor (100)
        """
        encoding = self.encoder(x, attention_mask=x_mask,
                                output_hidden_states=True)  # [last_states; pooler_hidden; all_hidden_states]
        encoding = self.sentence_hidden(encoding, feature_layers)  # batch_size * 768
        encoding = self.encoder_adaptor(encoding)  # batch_size *100
        encoding = self.dropout(encoding)
        encoding = torch.relu(encoding)
        return encoding
    
    def sentence_hidden(self, encoding, feature_layers):
        """
        How to generate representation from ptm representation
        Args:
            encoding (the output of pretrained_language model):
        Returns:
        
        """
        # todo: modify into prob_layer based analysis
        # encoding = torch.mean(encoding.last_hidden_state, dim=1)
        rep_ = None
        # Fixme
        if feature_layers is None:
            feature_layers = self.feature_layers # batch_size * 768

        rep_ = torch.stack([encoding.hidden_states[feature_layer] for feature_layer in feature_layers], dim=0)
        rep_ = torch.mean(rep_, dim=0)
        
        encoding = torch.mean(rep_, dim=1)
        
        return encoding
    
    def classify(self, feature: torch.Tensor, proto: torch.Tensor = None) -> torch.Tensor:
        if self.require_proto:
            if proto is not None:
                return feature * torch.transpose(proto, 0, 1)
            else:
                return feature * torch.transpose(self.proto, 0, 1)
        else:
            return self.classifier(feature)
    
    def layer_wise_forward(self, x: torch.Tensor, x_mask: torch.Tensor, feature_layers: [int]=None, proto: torch.Tensor = None, task_id=None) -> [
        torch.Tensor]:
        feature = self.features(x, x_mask, feature_layers)
        output = self.classify(feature, proto)  # -1 * dim_output
        
        layer_wise_feature = torch.cat([feature, output], dim=1)
        return layer_wise_feature
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,  feature_layers: [int]=None, proto: torch.Tensor = None,
                task_id=None) -> torch.Tensor:
        feature = self.features(x, x_mask, feature_layers)
        output = self.classify(feature, proto)  # -1 * dim_output
        return output
    
    
    
    
    # def predict(self, x: torch.Tensor, x_mask: torch.Tensor,  feature_layers: [int]=None, proto: torch.Tensor = None,
    #             task_id=None):
    #     print("PTM 189: ", "!"*50)
    #     if self.args.auto_layer:
    #         hs = self.encoder(x, attention_mask=x_mask, output_hidden_states=True)  # [last_states; pooler_hidden; all_hidden_states]
    #         if feature_layers is None:
    #             feature_layers = self.feature_layers
    #
    #         rep_ = torch.stack([hs.hidden_states[feature_layer] for feature_layer in feature_layers], dim=0)
    #         rep_mean = torch.mean(rep_, dim=0)  # batch_size * 768
    #
    #         encoding = torch.mean(rep_mean, dim=1)
    #         encoding = self.encoder_adaptor(encoding)  # batch_size *100
    #         encoding = self.dropout(encoding)
    #         encoding = torch.relu(encoding)
    #         output1 = self.classifier(encoding)
    #
    #         rep_middle = hs.hidden_states[feature_layers[0]]
    #         output2 = rep_middle * torch.transpose(self.proto, 0, 1)
    #
    #         output = output2 + output1
    #         return output
    #
    #     else:
    #         feature = self.features(x, x_mask, feature_layers)
    #         output = self.classify(feature, proto)  # -1 * dim_output
    #         return output
    #

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
        encoding = self.sentence_hidden(encoding, [prob_l])  # batch_size * 768
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
        encoding = self.sentence_hidden(encoding, [prob_l])  # batch_size * 768
        proto = torch.transpose(proto, 0, 1)
        output = torch.matmul(encoding, proto)
        return output
    
    def prob_final_classify(self, x: torch.Tensor, x_mask: torch.Tensor, prob_l: int):
        feature = self.features(x, x_mask, feature_layers=[prob_l])
        output = self.classify(feature)  # -1 * dim_output
        return output
    
    # parameter
    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor
        """
        params = []
        for pp in list(filter(lambda p: p.requires_grad, self.net.parameters())):
            if pp.grad is not None:
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
        
        for pp in list(filter(lambda p: p.requires_grad, self.net.parameters())):
            if pp.grad is not None:
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
    
    # freeze parameter
    def fix_parameter(self):
        if self.args.fix_layers == "all":
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        elif self.args.fix_layers == "none":
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True
        elif self.args.fix_layers == "bottom_n":
            n = self.args.fix_layers_n
            fix_layers = [i for i in range(self.encoder.config.num_hidden_layers)][:n]
            fix_layers = [f'layer.{i}' for i in fix_layers]
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True
                for ele in fix_layers:
                    if ele in name:
                        param.requires_grad = False
    
    # reset parameter
    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.encoder = self.ptm_.from_pretrained(supported_ptm[self.ptm][1], output_hidden_states=True)
        self.head.apply(xavier)
        self.fix_parameter()
