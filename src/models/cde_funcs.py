import math 
import torch 
from torch import nn
import torchcde

class CDEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(CDEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, latent_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):
        h1 = self.elu(self.lin1(x))
        h2 = self.relu(self.lin2(h1))
        out = self.lin3(h2)

        return out

class NeuralCDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.interpolation = interpolation

    def forward(self, z0, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' inteprolation methods are implemented")

        # Solve CDE
        zt = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)

        # Calculate predicted y
        zt = zt[:, 1]
        pred_y = self.readout(zt)
        return pred_y
        
