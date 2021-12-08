import math 
import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Variable

from .ode_funcs import NeuralODE, ODEFunc
from .utils import reparameterize

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

        # sample from initial encoding
        _, h0 = self.rnn(xt.flip((0,)).float())
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
        t = torch.squeeze(t)

        zs = self.ode(z0, t)
        hs = self.l2h(zs)
        xs = self.h2o(hs)

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
            z = reparameterize(mu, logvar)

        x_p = self.neural_decoder(z, t)
        return x_p, z, mu, logvar

def loss_function(x_p, x, z, mu, logvar):
    reconstruction_loss = 0.5 * ((x - x_p)**2).sum(-1).sum(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
    
    loss = reconstruction_loss + kl_loss
    loss = torch.mean(loss)

    return loss