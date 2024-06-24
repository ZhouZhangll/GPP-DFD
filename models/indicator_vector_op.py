# -*- coding:utf-8  -*-
import re

import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoConfig
from utils.utils import *
import os

class IndicatorLayer(nn.Module):
    def __init__(self, config,
                 max_seq_length=128,
                 start_sparsity = 0.0,
                 target_sparsity=0.0,
                 total_steps = 10,
                 pruning_type="structured_heads+structured_mlp", ):
        super(IndicatorLayer, self).__init__()
        self.config = config
        self.start_sparsity = start_sparsity
        self.current_sparsity = 0
        self.target_sparsity = target_sparsity
        self.max_seq_length = max_seq_length
        self.pruning_type = pruning_type
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.dim_per_head = self.hidden_size // self.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head = self.params_per_head_layer // self.num_attention_heads

        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + self.hidden_size + self.hidden_size * 4
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        self.prunable_model_size = 0
        self.total_steps = total_steps

        self.indicator = {}
        self.grad = {}
        self.types = []
        self.lamda = {"head": 0.5, "int": 0.5}
        self.parameters_per_dim = {}

        types = self.pruning_type.split("+")
        for type in types:
            self.initialize_one_module(type)

    def initialize_one_module(self, module_name):
        if module_name == "structured_mlp":
            self.int_indicator = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)
            self.indicator['int'] = self.int_indicator
            self.types.append('int')
            self.parameters_per_dim['int'] = self.params_per_intermediate_dim
            self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
        elif module_name == "structured_heads":
            self.head_indicator = self.initialize_parameters(self.num_attention_heads, self.num_hidden_layers)
            self.indicator['head'] = self.head_indicator
            self.types.append('head')
            self.parameters_per_dim['head'] = self.params_per_head
            self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads

    def initialize_parameters(self, size, num_layer):
            return nn.Parameter(torch.ones(num_layer, size))

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def TopKMask(self, grad, current_sparsity):
        l, h = grad.shape
        k = int(l * h * current_sparsity)
        sorted_weights, _ = torch.sort(torch.abs(grad).view(-1), descending=True)
        threshold = sorted_weights[k]
        mask = torch.gt(torch.abs(grad.view(-1)), threshold).float().view(l, h)
        return mask

    def get_current_sparsity(self, pruned_step):
        current_sparsity = np.square((1 - np.cos((np.pi * pruned_step)/self.total_steps))/2.0)*(self.target_sparsity-self.start_sparsity)+self.start_sparsity
        return current_sparsity

    def forward(self,pruned_step,grads):
        new_indicator = {f"{type}": [] for type in self.types}

        if self.total_steps > 0:
            self.current_sparsity = self.get_current_sparsity(pruned_step)
        else:
            return NotImplementedError

        for i, type in enumerate(self.types):
            if grads[type] is None:
                new_indicator[f"{type}"] = self.indicator[type]
            else:
                binary_indicator = self.TopKMask(grads[type]/pruned_step,self.current_sparsity)
                self.indicator[type].data.copy_(
                                         (1 - binary_indicator) + (1 - self.current_sparsity / self.target_sparsity) * binary_indicator)
                new_indicator[f"{type}"] = self.indicator[type]

        return new_indicator

    def get_indicator_loss(self):
        total_loss = 0
        for i, type in enumerate(self.types):
            total_loss +=  self.lamda[type] * torch.norm(self.indicator[type],p=1)

        return total_loss, self.current_sparsity



if __name__ == "__main__":
    config = AutoConfig.from_pretrained('bert-base-uncased')
    indicator = IndicatorLayer(config)