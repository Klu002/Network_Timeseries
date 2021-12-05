import math 
import torch 
<<<<<<< HEAD
from torch import nn
import torchcde

class CDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CDEFunc, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim * output_dim)
        self.relu = nn.RELU(inplace=True)

    def forward(self, x, t):

=======
import torchcde
>>>>>>> 3d9761497a59d6e7420d3201634b9d73756bb8ad
