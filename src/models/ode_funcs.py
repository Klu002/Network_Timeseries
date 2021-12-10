import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
import time

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim, time_invariant=False):
        super(ODEFunc, self).__init__()
        self.time_invariant = time_invariant

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        if time_invariant:
            self.lin1 = nn.Linear(latent_dim, hidden_dim)
        else:
            self.lin1 = nn.Linear(latent_dim + 1, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, latent_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h1 = self.elu(self.lin1(x))
        h2 = self.elu(self.lin2(h1))
        out = self.lin3(h2)

        return out

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, z0, t):
        # t is a 1d array with the timesteps

        # TODO: pass in t to ODE solver row by row then concatenate
        # output hidden states and pad
        pred_z = odeint(self.func, z0, t, method='dopri8')

        return pred_z