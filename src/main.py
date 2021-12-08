import math
import torch
import torchcde
import numpy as np

import os
import argparse
from models.vae import ODEVAE, loss_function
from data.preprocess import LoadInput, read_data, gen_batch

# TODO: Use cuda device instead of doing everything on CPU
def train(device, model, optimizer, loss_func, train_data, train_time, learning_rate, batch_size, epochs, n_sample, ckpt_path=None, use_cuda=False):  
  losses = []

  for epoch_idx in range(epochs):
      for i in range(train_data.shape[1]):
        try:
          batch_x, batch_t = gen_batch(train_data, train_time, i, i + batch_size, n_sample)

          optimizer.zero_grad()
          # if use_cuda:
          #     batch_x, batch_t = batch_x.cuda(), batch_t.cuda()

          x_p, z, z_mean, z_log_var = model(batch_x, batch_t)
          loss = loss_func(x_p, batch_x, z, z_mean, z_log_var)
          loss = torch.mean(loss)
          loss.backward()
          optimizer.step()
          losses.append(loss.item())

          print(f"Epoch {epoch_idx}")
          print(np.mean(losses), np.median(losses))

          if epoch_idx % 10 == 0 and ckpt_path:
            torch.save({
              'model_state_dict': model.state_dict(),
            }, ckpt_path + '_' + epoch_idx + '.pth')
            print('Saved model at {}'.format(ckpt_path + '_' + epoch_idx + '.pth'))

        except KeyboardInterrupt:
          return epoch_idx - 1


# class RunningAverageMeter(object):
#   """Computes and stores the average and current value"""

#   def __init__(self, momentum=0.99):
#     self.momentum = momentum
#     self.reset()

#   def reset(self):
#     self.val = None
#     self.avg = 0

#   def update(self, val):
#     if self.val is None:
#       self.avg = val
#     else:
#       self.avg = self.avg * self.momentum + val * (1 - self.momentum)
#     self.val = val

# def test(model, test_data):
#   test_X, test_Y = test_data
#   pred = model(test_X).squeeze(-1)
#   prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
#   proportion_correct = prediction_matches.sum() / test_y.size(0)

#   return proportion_correct

def main():
  parser = argparse.ArgumentParser(description='Latent ODE Model')
  parser.add_argument('--epochs', type=int, default=2000, help='Number of iterations to run model for')
  parser.add_argument('--lr', type=float, default=0.001, help='Starting learning rate')
  parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
  parser.add_argument('--n_sample', type=int, default=100, help='Number of time points to sample in each batch')

  parser.add_argument('--input', type=str, default=None, help='Path to csv file containing data')
  parser.add_argument('--load_dir', type=str, default='../data/processed', help='Path for loading data')
  parser.add_argument('--model_save_dir', type=str, default=None, help='Path for save checkpoints')
  parser.add_argument('--training_save_dir', type=str, default='../saved/training_weights', help='Path for saving model weights while training')
  parser.add_argument('--model_name', type=str, default='ODE', help='Name of model for save checkpoints')

  parser.add_argument('--visualize', type=eval, default=False)
  args = parser.parse_args()

  if args.input:
    print('Reading data from file...')
    df, dates = read_data(args.input)
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')

    # Saves data
    np.save('../data/processed/page_views.npy', df[dates].values)
    
  if args.load_dir:
    print('Loading data from file...')
    data_path = args.load_dir
    ld = LoadInput(data_path)
    train_data, val_data, test_data = ld.split_train_val_test()
    train_data, val_data, test_data = ld.load_data(train_data, val_data, test_data)
    train_time, val_time, test_time = ld.load_time(train_data, val_data, test_data)

    # Interpolation method to replace NaN values with numerical values
    ld.zero_interpolation(train_data, val_data, test_data)

  output_dim = 1
  hidden_dim = 64
  latent_dim = 6
  epochs = args.epochs
  lr = args.lr
  batch_size = args.batch_size
  n_sample = args.n_sample
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_data = train_data.to(device)
  val_data = val_data.to(device)
  test_data = test_data.to(device)
  train_time = train_time.to(device)
  val_time = val_time.to(device)
  test_time = test_time.to(device)

  model = ODEVAE(output_dim, hidden_dim, latent_dim).to(device)
  optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
  loss_func = loss_function
  # loss_meter = RunningAverageMeter()

  if args.save_dir:
    if not args.model_name:
      print('Please specify a model name to load')
    else:
      if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
      ckpt_path = os.path.join(args.save_dir, args.model_name + '.pth')

      if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_stat_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        train_data = checkpoint['train_data']
        val_data = checkpoint['val_data']
        test_data = checkpoint['test_data']
        train_time = checkpoint['train_time']
        val_time = checkpoint['val_time']
        test_time = checkpoint['test_time']
        # epochs = checkpoint['args'].epochs
        # lr = checkpoint['args'].lr
        # batch_size = checkpoint['args'].batch_size
        # n_sample = checkpoint['args'].n_sample

        print('Loaded checkpoint from {}'.format(ckpt_path))

  if args.save_dir and args.model_name:
    ckpt_path = os.path.join(args.train_dir, args.model_name)
    trained_epochs = train(device, model, optim, loss_func, train_data, train_time, lr, batch_size, epochs, n_sample, ckpt_path)

    print('Trained for {} epochs'.format(trained_epochs))
    if trained_epochs < epochs:
      torch.save({
        'model_state_dict': model.state_dict(),
      }, ckpt_path + '_' + trained_epochs + '.pth')
    else:
      ckpt_path = os.path.join(args.train_dir, args.model_name + '.pth')
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_time': train_time,
        'val_time': val_time,
        'test_time': test_time, 
        'args': args
      }, ckpt_path)
      print('Saved model at {}'.format(ckpt_path))
  else:
    train(device, model, optim, loss_func, train_data, train_time, lr, batch_size, epochs, n_sample)

  # TODO: if args.visualize, plot figures here

  
if __name__ == '__main__':
  main()