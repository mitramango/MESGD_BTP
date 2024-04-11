from typing import List
import torch
import copy
from copy import deepcopy
import transformers
import logging
import os

from torch.nn import Parameter
from utils import *

LOG = logging.getLogger(__name__)


class qa(torch.nn.Module):
    def __init__(self, model,model_tok,copy_of_state_dict):
        super(qa, self).__init__()
        
        self.model = model
        self.orig_state_dict = copy_of_state_dict
        '''Load Tokenizer
        '''
        self.model_tok = model_tok
        
        pass


    def optim_parameters(self):
        lora_params = self.model.parameters()
        return lora_params

    def edit(self, tokens, num_steps = 16, opt = "Adam"):
        # self.model = torch.nn.DataParallel(self.model)
        if opt == "Adam":
            optimizer = torch.optim.Adam(self.optim_parameters(), 1e-5)
        else:
            optimizer = torch.optim.SGD(self.optim_parameters(), 1e-5)
        
        self.losses = []
        for i in range(num_steps):
            outputs = self.model(**tokens)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            print(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
        self.loss = loss

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)



if __name__ == '__main__':
    pass


















