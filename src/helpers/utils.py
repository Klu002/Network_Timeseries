import torch
import os

def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    std = torch.exp(0.5*logvar)
    
    return mu + eps*std

def call_gru_d(gru_d, input):
    """
    Wrapper funtion to call GRUD_Cell. Puts inputs passed into standard GRU of size (num_time, batch_size, 2)
    in valid form to be passed into GRUD 
    """
    input = torch.permute(input, (1, 2, 0))

    gru_input = []
    for i in range(input.shape[0]):
        X = input[i, :-1, :]
        m = torch.where(torch.isnan(X), 0, 1)
        time = input[i,-1, :]
        delta = torch.zeros(X.shape)

        for t in range(input.shape[2]):
            #delta[t] = t_diff[t] if m[t] == 1 else delta[t-1] + t_diff[t]
            delta[:,t] = time[t] + (m[:, t] - 1) * delta[:, t-1]
        
        stacked = torch.stack([X, m, delta])
        gru_input.append(stacked)

    #gruD input size = (batch_size, 3, 2, num_time)
    #gruD input = (batch_size, (X, m, delta))
    gru_input = torch.stack(gru_input)
    return gru_d(gru_input)

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
