import torch
import os

def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    std = torch.exp(0.5*logvar)
    
    return mu + eps*std

# def save_checkpoint(state, save, epoch):
#     if not os.path.exists(save):
#         os.makedirs(save)
#     filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
#     torch.save(state, filename)

# def load_checkpoint(ckpt_path, model, device):
#     if not os.path.exists(ckpt_path):
#         raise Exception("Checkpoint " + ckpt_path + " does not exist.")
#     # Load checkpoint
#     checkpt = torch.load(ckpt_path)
#     ckpt_args = checkpt[]
