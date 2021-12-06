from _typeshed import Self
import math 
import numpy as np
from models.cde_funcs import ODEFunc

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
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

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
        z0 = self.hid2lat(h0[0])
        mu = z0[:, :self.latent_dim]
        logvar = z0[:, self.latent_dim:]
        return mu, logvar

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim 
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = ODEFunc(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        """"
            z0: 
            t: number of timesteps
        """
        zs = self.ode(z0, t)
        hs = self.l2h(zs)
        xs = Self.h2o(hs)

        return xs

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
        return x_p, z, mu, logvar
