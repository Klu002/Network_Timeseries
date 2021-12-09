import math

from numpy.core.numeric import NaN
import torch
import torchcde
import numpy as np

import os
import argparse
import time
from models.vae import ODEVAE, train_smape_loss, train_mae_loss, test_smape_loss, vae_loss_function
from data.preprocess import LoadInput, read_data, gen_batch, load_median_interpolation, load_time

import matplotlib.pyplot as plt

np.set_printoptions(threshold=500)

def plot_versus(x, x_p, x_start, x_end, save_path):
  plt.plot(x, label='True')
  plt.plot(x_p, label='Pred')
  plt.xticks(np.arange(x_start, x_end, 25))
  plt.legend()
  plt.savefig(save_path)
  plt.clf()

# TODO: Use cuda device instead of doing everything on CPU
def train(device, model_name, model, optimizer, train_loss_func, test_loss_func, train_data, train_time, learning_rate, batch_size, epoch_idx, epochs, n_sample, ckpt_path=None, use_cuda=False):
  epoch_losses = []
  for epoch_idx in range(epoch_idx, epochs):
    losses = []
    num_batches = math.ceil(train_data.shape[1]/batch_size)
    print("Num batches: {}\n".format(num_batches))

    batch_order_permutation = np.random.permutation(train_data.shape[1])
    np.random.shuffle(batch_order_permutation)

    for i in range(num_batches):
      try:
        start_time = time.time()
        optimizer.zero_grad()

        if i == num_batches - 1:
          batch_indices = batch_order_permutation[i*batch_size:]
        else:
          batch_indices = batch_order_permutation[i*batch_size:(i+1)*batch_size]
        batch_x, batch_t = gen_batch(train_data, train_time, batch_indices, n_sample)
        # print("Batch shape: ", batch_x.shape, batch_t.shape)

        max_len = np.random.randint(batch_t.shape[1]//2, batch_t.shape[1])
        permutation = np.random.permutation(batch_t.shape[0])
        np.random.shuffle(permutation)
        permutation = np.sort(permutation[:max_len])

        batch_x, batch_t = batch_x[permutation], batch_t[permutation]
        batch_x, batch_t = batch_x.to(device), batch_t.to(device)
        
        b = time.time()
        print("before model")
        x_p, z, z_mean, z_log_var = model(batch_x, batch_t, batch_t)
        x_p, z, z_mean, z_log_var = x_p.to(device), z.to(device), z_mean.to(device), z_log_var.to(device)
        x_p = torch.round(x_p)
        x_p[x_p < 0] = 0

        e = time.time()
        print("after model")
        print("Time taken model: {}".format(round(e - b, 3)))

        if i % 20 == 0:
          batch_x_plot = np.squeeze(batch_x.detach().cpu().numpy(), 2)
          x_p_plot = np.squeeze(x_p.detach().cpu().numpy(), 2)
          x_values = np.squeeze(batch_t.detach().cpu().numpy(), 2)
          x_start = x_values[:, 0][0]
          x_end = x_values[:, 0][-1]

          plot_versus(batch_x_plot[:, 0], x_p_plot[:, 0], x_start, x_end, '../saved/images/{}_epoch_{}_batch_{}'.format(model_name, epoch_idx, i))

        # with np.printoptions(threshold=50):
        #   print("True x: ", batch_x)
        #   print("Pred x: ", x_p)
        # If loss function = SMAPE, don't have to divide by max_len. 
        # If loss function = VAE_loss, must divide by max len.
        b = time.time()
        print("before loss calculation")
        mae_loss = train_loss_func(device, batch_x, x_p)
        kaggle_smape_loss = test_loss_func(device, batch_x, x_p)
        e = time.time()
        print("after loss calculation")
        print("Time taken loss: {}".format(round(e - b, 3)))
        # loss = loss_func(x_p, batch_x, z, z_mean, z_log_var)
        # loss /= max_len
        b = time.time()
        print("before loss backprop")
        mae_loss.backward()
        optimizer.step()
        e = time.time()
        print("after loss backprop")
        print("Time taken backprop: {}".format(round(e - b, 3)))
        
        losses.append(mae_loss.item())

        end_time = time.time()
        time_taken = end_time - start_time

        print("Batch {}/{}".format(i + 1, num_batches))
        print("{}s - mae: {} - kaggle_smape: {}".format(round(time_taken, 3), round(mae_loss.item(), 3), round(kaggle_smape_loss.item(), 3)))

      except KeyboardInterrupt:
        return epoch_idx, losses

    epoch_losses.append(losses)

    if epoch_idx > 0 and epoch_idx % 1 == 0 and ckpt_path:
      torch.save({
        'model_state_dict': model.state_dict(),
        'epoch_idx': epoch_idx,
        'num_epochs': epochs,
        'losses': epoch_losses,
      }, ckpt_path + '_' + str(epoch_idx + 1) + '.pth')
      print('Saved model at {}'.format(ckpt_path + '_' + str(epoch_idx + 1) + '.pth'))

    print("Epoch {}/{}".format(epoch_idx + 1, epochs))
    print("mean differentiable_smape: {} - median differentiable_smape: {}\n".format(np.mean(losses), np.median(losses)))

  return epoch_idx + 1, epoch_losses

# def test(device, model, optimizer, test_loss_func, test_data, test_time, batch_size, n_sample, use_cuda=False):
#   num_batches = math.ceil(test_data.shape[1]/batch_size)
#   print("Num batches: {}\n".format(num_batches))

#   losses = []
#   for i in range(num_batches):
#     start_time = time.time()
#     optimizer.zero_grad()

#     if i == num_batches - 1:
#       batch_indices = np.arange(i * batch_size, test_data.shape[1])
#     else:
#       batch_indices = np.arange(i * batch_size, (i + 1) * batch_size)
#     batch_x, batch_t = gen_batch(test_data, test_time, batch_indices, n_sample)
#     batch_x, batch_t = batch_x.to(device), batch_t.to(device)

#     x_p, z, _, _ = model(batch_x, batch_t)
#     x_p, z = x_p.to(device), z.to(device)

#     kaggle_smape_loss = test_loss_func(device, batch_x, x_p)
#     losses.append(kaggle_smape_loss.item())

#     end_time = time.time()
#     time_taken = end_time - start_time

#     print("Batch {}/{}".format(i + 1, num_batches))
#     print("{}s - kaggle_smape: {}".format(round(time_taken, 3), round(kaggle_smape_loss.item(), 3)))

#   print("Avg Test Loss - {}".format(np.mean(losses)))

def main():
  parser = argparse.ArgumentParser(description='Latent ODE Model')
  parser.add_argument('--epochs', type=int, default=2000, help='Number of iterations to run model for')
  parser.add_argument('--lr', type=float, default=0.001, help='Starting learning rate')
  parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
  parser.add_argument('--n_sample', type=int, default=100, help='Number of time points to sample in each batch')

  parser.add_argument('--input', type=str, default=None, help='Path to csv file containing data')
  parser.add_argument('--load_dir', type=str, default='../data/processed', help='Path for loading data')
  parser.add_argument('--model_save_dir', type=str, default=None, help='Path for save checkpoints')
  parser.add_argument('--training_save_dir', type=str, default=None, help='Path for saving model weights while training')
  parser.add_argument('--model_name', type=str, default='ODE', help='Name of model for save checkpoints')

  parser.add_argument('--encoder', type=str, default='gru')
  parser.add_argument('--use_cuda', type=eval, default=False)
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
    train_data, _, _ = ld.split_train_val_test(1, 0, 0)
    train_data, _, _ = load_median_interpolation(train_data, None, None)
    train_time, _, _ = load_time(train_data, None, None)

  output_dim = 1
  hidden_dim = 64
  latent_dim = 6
  epoch_idx = 0
  epochs = args.epochs
  lr = args.lr
  batch_size = args.batch_size
  n_sample = args.n_sample
  device = 'cpu'

  if args.use_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("Using device: ", device)

  train_data = train_data.to(device)
  # val_data = val_data.to(device)
  # test_data = test_data.to(device) train_time = train_time.to(device)
  # val_time = val_time.to(device)
  # test_time = test_time.to(device)

  model = ODEVAE(output_dim, hidden_dim, latent_dim, encoder=args.encoder).to(device)
  optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
  train_loss_func = train_mae_loss
  test_loss_func = test_smape_loss
  # loss_func = vae_loss_function
  # loss_meter = RunningAverageMeter()

  if args.model_save_dir:
    if not args.model_name:
      print('Please specify a model name to load')
    else:
      if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
      ckpt_path = os.path.join(args.model_save_dir, args.model_name + '.pth')

      if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch_idx' in checkpoint:
          epoch_idx = checkpoint['epoch_idx']
        # epochs = checkpoint['args'].epochs
        # lr = checkpoint['args'].lr
        # batch_size = checkpoint['args'].batch_size
        # n_sample = checkpoint['args'].n_sample
        last_epoch_saved = ckpt_path[:-4].rsplit("_", 1)
        if last_epoch_saved[1].isnumeric():
          args.model_name = last_epoch_saved[0].rsplit("/", 1)[1]
          print("Model name changed to: ", args.model_name)
        print('Loaded checkpoint from {}'.format(ckpt_path))

  # orig_trajs, samp_trajs, samp_ts = generate_spirals()

  done_training = True
  if args.training_save_dir and args.model_name:
    ckpt_path = os.path.join(args.training_save_dir, args.model_name)
    trained_epochs, losses = train(device, args.model_name, model, optim, train_loss_func, test_loss_func, train_data, train_time, lr, batch_size, epoch_idx, epochs, n_sample, ckpt_path)

    print('Trained for {} epochs'.format(trained_epochs))
    if trained_epochs > 0 and trained_epochs < epochs:
      torch.save({
        'model_state_dict': model.state_dict(),
        'epoch_idx': trained_epochs - 1,
        'num_epochs': epochs,
        'losses': losses
      }, ckpt_path + '_' + str(trained_epochs) + '.pth')
      print('Interrupted model training - saved checkpoint to {}'.format(ckpt_path))
      
    if trained_epochs < epochs:
      done_training = False
  else:
    trained_epochs, losses = train(device, args.model_name, model, optim, train_loss_func, test_loss_func, train_data, train_time, lr, batch_size, epoch_idx, epochs, n_sample)
    if trained_epochs < epochs:
      done_training = False

  if done_training:
    final_model_path = os.path.join(args.model_save_dir, args.model_name + '.pth')
    torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optim.state_dict(),
      'train_data': train_data,
      # 'val_data': val_data,
      # 'test_data': test_data,
      'train_time': train_time,
      # 'val_time': val_time,
      # 'test_time': test_time, 
      'losses': losses,
      'args': args
    }, final_model_path)
    print('Saved model at {}'.format(final_model_path))


  # TODO: if args.visualize, plot figures here


if __name__ == '__main__':
  main()
