import torch
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
from GRUD_layer import GRUD_cell

class grud_model(torch.nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers = 1, x_mean = 0,\
     bias =True, batch_first = False, bidirectional = False, dropout_type ='mloss', dropout = 0):
        super(grud_model, self).__init__()

        self.gru_d = GRUD_cell(input_size = input_size, hidden_size= hidden_size, output_size=output_size, 
                dropout=dropout, dropout_type=dropout_type, x_mean=x_mean)
        self.hidden_to_output = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.num_layers >1:
            #(batch, seq, feature)
            self.gru_layers = torch.nn.GRU(input_size = hidden_size, hidden_size = hidden_size, batch_first = True, num_layers = self.num_layers -1, dropout=dropout)

    def initialize_hidden(self, batch_size):
        device = next(self.parameters()).device
        # The hidden state at the start are all zeros
        return torch.zeros(self.num_layers-1, batch_size, self.hidden_size, device=device)

    def forward(self,input):
        output, hidden = self.gru_d(input)
        # batch_size, n_hidden, n_timesteps

        if self.num_layers >1:
            #TODO remove init hidden, not necessary, auto init works fine
            init_hidden = self.initialize_hidden(hidden.size()[0])
            

            output, hidden = self.gru_layers(hidden)#, init_hidden)
  

            output = self.hidden_to_output(output)
            output = torch.sigmoid(output)

        #print("final output size passed as model result")
        #print(output.size())
        return output