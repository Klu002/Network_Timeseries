import math 
import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Variable

# from models.ode_funcs import NeuralODE, ODEFunc
from models.ode_funcs import ODEFunc, NeuralODE
from models.spirals import NNODEF
from helpers.utils import reparameterize

np.set_printoptions(threshold=500)

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
        
        # func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        func = ODEFunc(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        """"
            z0: 
            t: number of timesteps
        """
        t_1d = t[:, 0, 0]

        zs = self.ode(z0, t_1d)
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

        # x_p = self.neural_decoder(z, t).permute(1, 0, 2)
        x_p = self.neural_decoder(z, t)
        return x_p, z, mu, logvar

def vae_loss_function(x_p, x, z, mu, logvar):
    reconstruction_loss = 0.5 * ((x - x_p)**2).sum(-1).sum(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), -1)
    
    loss = reconstruction_loss + kl_loss
    loss = torch.mean(loss)

    return loss

def differentiable_smape(device, y_true, y_pred, mask, epsilon=0.1):
    constant_and_epsilon = torch.tensor(0.5 + epsilon).repeat(y_true.shape).to(device)
    summ = torch.maximum(torch.abs(y_true) + torch.abs(y_pred) + epsilon, constant_and_epsilon)
    smape = torch.abs(y_pred - y_true) / summ

    nvalid = torch.sum(mask)
    smape_sum = torch.sum(smape * mask) / nvalid
    return smape_sum

def rounded_smape(device, y_true, y_pred, mask):
    y_true_copy = torch.round(y_true).type(torch.IntTensor)
    y_pred_copy = torch.round(y_pred).type(torch.IntTensor)
    summ = torch.abs(y_true) + torch.abs(y_pred)
    smape = torch.where(summ == 0, torch.zeros_like(summ), torch.abs(y_pred_copy - y_true_copy) / summ)
    
    nvalid = torch.sum(mask)
    smape_sum = torch.sum(smape * mask) / nvalid
    return smape_sum

def kaggle_smape(device, y_true, y_pred, mask):
    summ = torch.abs(y_true) + torch.abs(y_pred)
    smape = torch.where(summ == 0, torch.zeros_like(summ), torch.abs(y_pred - y_true) / (2 * summ))

    nvalid = torch.sum(mask)
    smape_sum = (100 * torch.sum(smape * mask)) / nvalid

    return smape_sum

def mae(device, y_true, y_pred, mask):
    y_true_log = torch.log1p(y_true)
    y_pred_log = torch.log1p(y_pred)
    error = torch.abs(y_true_log - y_pred_log)/2

    nvalid = torch.sum(mask)
    error_sum = torch.sum(error * mask) / nvalid
    return error_sum

def train_smape_loss(device, y_true, y_pred):
    mask = torch.isfinite(y_true).to(device)
    weight_mask = mask.type(torch.FloatTensor)
    
    return differentiable_smape(device, y_true, y_pred, weight_mask)

def test_smape_loss(device, y_true, y_pred):
    mask = torch.isfinite(y_true).to(device)
    weight_mask = mask.type(torch.FloatTensor)
    
    return kaggle_smape(device, y_true, y_pred, weight_mask)
