import math
import torch
import torchcde
import numpy as np

import os
import argparse

from preprocess import LoadInput, read_data, gen_batch

def train(use_cuda, orig_trajs, samp_trajs, samp_ts):
  vae = ODEVAE(2, 64, 6)
  # vae = vae.cuda()
  # if use_cuda:
  #     vae = vae.cuda()
  optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)
  preload = False
  n_epochs = 20000
  batch_size = 100

  plot_traj_idx = 1
  plot_traj = orig_trajs[:, plot_traj_idx:plot_traj_idx+1]
  plot_obs = samp_trajs[:, plot_traj_idx:plot_traj_idx+1]
  plot_ts = samp_ts[:, plot_traj_idx:plot_traj_idx+1]
  if use_cuda:
      plot_traj = plot_traj.cuda()
      plot_obs = plot_obs.cuda()
      plot_ts = plot_ts.cuda()

  if preload:
      vae.load_state_dict(torch.load("models/vae_spirals.sd"))

  for epoch_idx in range(n_epochs):
      losses = []
      train_iter = gen_batch(batch_size)
      for x, t in train_iter:
          optim.zero_grad()
          if use_cuda:
              x, t = x.cuda(), t.cuda()

          max_len = np.random.choice([30, 50, 100])
          permutation = np.random.permutation(t.shape[0])
          np.random.shuffle(permutation)
          permutation = np.sort(permutation[:max_len])

          x, t = x[permutation], t[permutation]

          x_p, z, z_mean, z_log_var = vae(x, t)
          kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
          loss = 0.5 * ((x-x_p)**2).sum(-1).sum(0) / noise_std**2 + kl_loss
          loss = torch.mean(loss)
          loss /= max_len
          loss.backward()
          optim.step()
          losses.append(loss.item())

      print(f"Epoch {epoch_idx}")

      frm, to, to_seed = 0, 200, 50
      seed_trajs = samp_trajs[frm:to_seed]
      ts = samp_ts[frm:to]
      if use_cuda:
          seed_trajs = seed_trajs.cuda()
          ts = ts.cuda()

      samp_trajs_p = to_np(vae.generate_with_seed(seed_trajs, ts))

      #fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))
      #axes = axes.flatten()
      #for i, ax in enumerate(axes):
      #    ax.scatter(to_np(seed_trajs[:, i, 0]), to_np(seed_trajs[:, i, 1]), c=to_np(ts[frm:to_seed, i, 0]), cmap=cm.plasma)
      #    ax.plot(to_np(orig_trajs[frm:to, i, 0]), to_np(orig_trajs[frm:to, i, 1]))
      #    ax.plot(samp_trajs_p[:, i, 0], samp_trajs_p[:, i, 1])
      #plt.show()

      print(np.mean(losses), np.median(losses))
      clear_output(wait=True)

"""
def train(model, num_epochs, train_data, optimizer, loss_func):
  train_X, train_Y = train_data

  train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

  for epoch in range(num_epochs):
      for batch in train_dataloader:
          batch_coeffs, batch_y = batch
          pred_y = model(batch_coeffs).squeeze(-1)
          loss = loss_func(pred_y, batch_y)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))
"""

def test(model, test_data):
  test_X, test_Y = test_data
  pred = model(test_X).squeeze(-1)
  prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
  proportion_correct = prediction_matches.sum() / test_y.size(0)

  return proportion_correct

def run():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--load', type=str, help='Load data from file')
    args = parser.parse_args()
  
    if args.input:
        print('Reading data from file')
        data_path = args.input
        df, dates = read_data(data_path)
        if not os.path.exists('data/parsed'):
            os.makedirs('data/parsed')

        # Saves data
        np.save('data/parsed/page_views.npy', df[dates].values)
    
    if args.load:
        print('Loading data from file')
        data_path = args.load
        ld = LoadInput(data_path)
        train_data, val_data, test_data = ld.split_train_val_test()
        train_data, val_data, test_data = ld.load_data(train_data, val_data, test_data)
        train_time, val_time, test_time = ld.load_time(train_data, val_data, test_data)
        return train_data, val_data, test_data, train_time, val_time, test_time
    

def main():
  parser = argparse.ArgumetnParser(description='Latent ODE Model')
  parser.add_argument('--niters', type=int, default=2000, help='Number of iterations to run model for')
  parser.add_argument('--lr', type=float, default=0.01, help='Starting learning rate')
  parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')

  parser.add_argument('--input', type=str, default='./data/train_2.csv', help='Input file')
  parser.add_argument('--load_dir', type=str, default='./saved/data', help='Path for loading data')
  parser.add_argument('--save_dir', type=str, default='./saved/models', help='Path for save checkpoints')

  args = parser.parse_args()

  if args.input:
        print('Reading data from file')
        data_path = args.input
        df, dates = read_data(data_path)
        if not os.path.exists('saved/data/parsed'):
            os.makedirs('saved/data/parsed')

        # Saves data
        np.save('saved/data/parsed/page_views.npy', df[dates].values)
    
  if args.load:
      print('Loading data from file')
      data_path = args.load
      ld = LoadInput(data_path)
      train_data, val_data, test_data = ld.split_train_val_test()
      train_data, val_data, test_data = ld.load_data(train_data, val_data, test_data)
      train_time, val_time, test_time = ld.load_time(train_data, val_data, test_data)
      return train_data, val_data, test_data, train_time, val_time, test_time


  #Hyperparameters
  num_epochs = 30
  #loss_func =

  #TODO: Get data here
  #test_data, train_data = #[FILL IN HERE]

  #Initialize Model
  model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)
  
  optimizer = torch.optim.Adam(model.parameters())

  #Train and Test
  train(model, num_epochs, train_data, optimizer, loss_func)
  acc = test(model, test_data)

  print('Test Accuracy: {}'.format(proportion_correct))

if __name__ == '__main__':
  main()
