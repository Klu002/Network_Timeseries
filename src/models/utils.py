import torch

def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    std = torch.exp(0.5*logvar)
    return mu + eps*std