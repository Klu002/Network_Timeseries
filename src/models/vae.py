import math 
import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import Functional as F
from torch.autograd import Variable

from utils import reparameterize

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, t):
        t = t.clone()
        # 1st element is t = 0, remainder are negative offset from that
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.

        # concatenate input
        xt = torch.cat((x, t), dim=-1)
        xt = xt[::-1]

        # sample from initial encoding
        _, h0 = self.rnn(xt)
        mu = self.mu(h0[0])
        logvar = self.logvar(h0[0])

        return mu, logvar

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim 
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.rnn_encoder = RNNEncoder(output_dim, hidden_dim, latent_dim)
        self.neural_decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)
        
    def forward(self, x, t, MAP=False):
        mu, logvar = self.rnn_encoder(x, t)
        if MAP:
            z = mu
        else:
            z = reparametrize(mu, logvar)

        x_p = self.decoder(z, t)
        return mu, logvar, x_p
