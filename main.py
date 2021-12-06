import math
import torch
import torchcde
import numpy as np

import os
import argparse
from models.vae import ODEVAE
from preprocess import LoadInput, read_data, gen_batch

def train(model, loss_func, use_cuda=False):
  vae = model
  optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)
  n_epochs = 200

  for epoch_idx in range(n_epochs):
      losses = []

      optim.zero_grad()
      if use_cuda:
          x, t = x.cuda(), t.cuda()

      x_p, z, z_mean, z_log_var = vae(x, t)
      loss = loss_func(z_log_var, z_mean, z_log_var)
      loss = torch.mean(loss)
      loss.backward()
      optim.step()
      losses.append(loss.item())

      print(f"Epoch {epoch_idx}")
      print(np.mean(losses), np.median(losses))

def test(model, test_data):
  test_X, test_Y = test_data
  pred = model(test_X).squeeze(-1)
  prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
  proportion_correct = prediction_matches.sum() / test_y.size(0)

  return proportion_correct

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

  model = ODEVAE(2, 64, 6)
  for i in range(train_data.shape[1]):
    site = train_data[:, i, :]
    train(model, site)

  
if __name__ == '__main__':
  main()
