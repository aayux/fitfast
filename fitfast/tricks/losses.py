import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OHEMLoss(torch.nn.NLLLoss):
    r""" Online hard example mining loss function.
    """
    def __init__(self, ratio=.2, weight=None):
        super().__init__(weight, True)
        self.ratio = ratio
        # self.weight = weight
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, input, target,ratio=None):
        if ratio is not None:
            self.ratio = ratio 
        
        n_instances = input.size(0)
        n_hardex = int(self.ratio * n_instances)
        
        input = self.log_softmax(input)
        x = input.clone()
        per_exloss = torch.autograd.Variable(torch.zeros(n_instances)).cuda()
        
        for idx, label in enumerate(target.data):
            per_exloss[idx] = - x.data[idx, label]
        
        _, idx = per_exloss.topk(n_hardex)
        x = input.index_select(0, idx)
        y = target.index_select(0, idx)
        # w = self.weight.index_select(0, idx)
        
        return F.nll_loss(x, y)

