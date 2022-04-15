import torch
import random
from utils.per_buffer_NLP import PERBufferNLP
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from datasets.transforms.LangevinDynamic_aug import InstanceLangevinDynamicAug
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Experience Replay for NLP.')
    # add related arguments
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    
    parser.add_argument('--auto_layer', required=False, action="store_true")
    parser.add_argument('--auto_feature', required=False, action="store_true")
    parser.add_argument('--auto_current_task', required=False, action="store_true")
    
    return parser


class AutoVanilla3NLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'autovanilla3nlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(AutoVanilla3NLP, self).__init__(backbone, loss, args, transform)
        
        self.buffer = PERBufferNLP(self.args.buffer_size, self.device, require_label_name=False, PTM=self.args.ptm)
        self.candidate_feature_layers = [i for i in range(self.net.encoder.config.num_hidden_layers)][-6:]
        self.current_feature_layers = args.feature_layers
        self.current_task = 0

        self.proto_layers = [8, 9]

        self.auto_layer = args.auto_layer
        self.auto_feature = args.auto_feature
        self.auto_current_task = args.auto_current_task
        self.threshold = 0.8
    
    def end_task(self, dataset):
        self.current_task += 1
    
    def begin_task(self, dataset):
        print(f"current feature layers: {self.current_feature_layers}")
        print(f"candidate feature layers: {self.candidate_feature_layers}")
        self.net.eval()
        if self.current_task == 0 or len(self.candidate_feature_layers) == 1 or not self.auto_layer:
            pass
        else:
            toss = random.random()
            print("toss: ", toss)
            if toss > 0.7:
                acc = self._evaluate_layer(probing_batch_num=10, dataset=dataset)
                print("in-job probing acc: ", acc)
                selected_feature_layer = torch.where(acc > self.threshold, torch.tensor(True).to(self.device),
                                                     torch.tensor(False).to(self.device))
                candidate_feature_layers = [self.candidate_feature_layers[i] for i, selected in
                                            enumerate(selected_feature_layer) if selected]
                
                if len(candidate_feature_layers) != 0:
                    # max_layer = torch.argmax(acc).item()
                    layer_id = random.randint(0, len(candidate_feature_layers) - 1)
                    max_layer = candidate_feature_layers[layer_id]
                else:
                    max_layer = torch.argmax(acc, dim=0).item()
                    max_layer = self.candidate_feature_layers[max_layer]
                
                feature_layers = [max_layer]
                
                if len(candidate_feature_layers) == 0 and len(self.candidate_feature_layers) > 3:
                    min_layer = torch.argmin(acc).item()
                    # self.candidate_feature_layers.remove(min_layer)
                    self.threshold = max(0.5, self.threshold - 0.05)
                
                feature_layers.append(self.current_feature_layers[-1])
                # the current best layer and the last best layer
                
                # self.current_feature_layers = feature_layers
                
                print("feature_layers: ", feature_layers)
                self.net.feature_layers = feature_layers
        
        self.net.train()
    
    def _collect_encodings(self, probing_batch_num: int, dataset):
        labels = []
        encodings = []
        
        for i in range(probing_batch_num):
            (m_examples, m_examples_mask, m_labels, m_features), choice = self.buffer.get_data(
                self.args.minibatch_size)
            
            labels.append(m_labels)
            hidden = self.net.encoder(m_examples, m_examples_mask)
            rep_ = torch.stack([hidden.hidden_states[feature_layer] for feature_layer in self.candidate_feature_layers],
                               dim=0)
            encoding = torch.mean(rep_, dim=2)  # feature_layer_num, batch_size, embedding_dim
            encoding = torch.transpose(encoding, 0, 1)  # batch_size, feature_layer_num, embedding_dim
            encodings.append(encoding)
        
        if self.auto_current_task:
            for i, data in enumerate(dataset.train_loader):
                _, _, x_token_idxs, x_token_masks, _, _, y_idxs = data
                
                x_token_idxs = x_token_idxs.to(self.device)
                x_token_masks = x_token_masks.to(self.device)
                y_idxs = y_idxs.to(self.device)
                
                labels.append(y_idxs)
                
                hidden = self.net.encoder(x_token_idxs, x_token_masks)
                rep_ = torch.stack(
                    [hidden.hidden_states[feature_layer] for feature_layer in self.candidate_feature_layers],
                    dim=0)
                encoding = torch.mean(rep_, dim=2)  # feature_layer_num, batch_size, embedding_dim
                encoding = torch.transpose(encoding, 0, 1)  # batch_size, feature_layer_num, embedding_dim
                encodings.append(encoding)
                
                if i > min(probing_batch_num, 4):
                    break
        
        return encodings, labels
    
    def _evaluate_layer(self, probing_batch_num: int, dataset):
        feature_layer_num = len(self.candidate_feature_layers)
        
        encoding_list, label_list = self._collect_encodings(probing_batch_num, dataset)
        
        encodings = torch.cat(encoding_list, dim=0)  # example_size, feature_layer_num, embed_dim
        labels = torch.cat(label_list, dim=0)  # example_size
        example_num = encodings.shape[0]
        
        masks = []
        for i in torch.unique(labels):
            mask = torch.where(labels == i, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
            count = torch.sum(mask, dim=0)
            mask = torch.div(mask, count)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=0)  # class_num, example_size
        encodings = torch.transpose(encodings, 0, 2)  # embed_dim, feature_layer_num, example_size
        encodings = torch.unsqueeze(encodings, dim=2)  # embed_dim, feature_layer_num, 1, example_size
        
        prototypes = torch.mul(masks, encodings)  # embed_dim, feature_layer_num, class_num, example_size
        prototypes = torch.mean(prototypes, dim=3)  # embed_dim, feature_layer_num, class_num
        prototypes = torch.transpose(prototypes, 0, 2)  # class_num, feature_layer_num, embed_dim
        
        acc_list = []
        for i, (encs, ls) in enumerate(zip(encoding_list, label_list)):
            encs = torch.unsqueeze(encs, dim=1)  # example_size, 1, feature_layer_num, embed_dim
            
            dist = torch.mul(encs, prototypes)  # example_size, class_num, feature_layer_num, embed_dim
            dist = torch.sum(dist, dim=-1)  # example_size, class_num, feature_layer_num
            dist = torch.sigmoid(dist)  # example_size, class_num, feature_layer_num
            dist = torch.transpose(dist, 1, 2)  # example_size, feature_layer_num, class_num
            predict = torch.argmax(dist, dim=2)  # example_size, feature_layer_num
            
            ls = torch.unsqueeze(ls, dim=1)
            ls = ls.expand(-1, feature_layer_num)
            
            acc_ = torch.where(predict == ls, torch.tensor(1).to(self.device),
                               torch.tensor(0).to(self.device))  # example_size, feature_layer_num
            acc_ = torch.sum(acc_, dim=0)  # feature_layer_num
            acc_list.append(acc_)
        
        acc_count = torch.stack(acc_list, dim=0)
        print("acc_count 169", acc_count)
        acc_count = torch.sum(acc_count, dim=0)
        print("acc_count 170", acc_count)
        acc_count = torch.div(acc_count, example_num)  # feature_layer_num
        return acc_count
    
    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask, self.current_feature_layers)
        loss = self.loss(outputs, labels)
        # end: Loss 1

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        encoding = self.net.encoder(inputs, attention_mask=inputs_mask,output_hidden_states=True)
        features = self.net.sentence_hidden(encoding, self.current_feature_layers)  # batch_size * 768
        self.buffer.add_data(examples=inputs,
                             examples_mask=inputs_mask,
                             labels=labels,
                             features=features.data,
                             task_labels=task_labels,
                             labels_name=labels_name,
                             labels_name_mask=labels_mask)
        
        return loss.item()
    
    def describe(self, buf_inputs):
        """
        describe the basic information of buffer_inputs
        Parameters
        ----------
        buf_inputs : data/tensor

        Returns None
        -------
        """
        print(buf_inputs.type())  # 数据类型
        print(buf_inputs.size())  # 张量的shape，是个元组
        print(buf_inputs.dim())  # 维度的数量
