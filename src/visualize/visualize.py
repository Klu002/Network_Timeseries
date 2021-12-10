import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from models.vae import mape, differentiable_smape, rounded_smape, kaggle_smape, mae, mse
import os
import torch
import pandas as pd

# sns.set_theme(style="darkgrid")

def plot_real_vs_pred(x, x_p, x_start, x_end, save_path):
  plt.plot(x, label='True')
  plt.plot(x_p, label='Pred')
  plt.xticks(np.arange(x_start, x_end, 25))
  plt.legend()
  plt.savefig(save_path)
  plt.clf()

# TODO: Add way to visualize the loss functions against each other
# (both near 0 and in general)
def visualize_loss(y_true, y_pred, x, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Behavior of Loss Functions')

    ax1.plot(x, smape, color='blue', label='SMAPE')
    ax1.plot(x, diff_smape, color='red', label='Differentiable SMAPE')
    ax1.plot(x, rounded_smape, color='green', label='Rounded SMAPE')
    ax1.plot(x, mape, color='yellow', lable='MAPE')
    ax1.plot(x, mae_log, color='magenta', label='MAE_Log')
    ax1.plot(x, mse_log, color='black', label='MSE_Log')
    
    ax1.legend()
    plt.savefig(save_path)

def visualize_loss_real_results(y_true, y_pred, x, save_path):
    pass

# TODO: Add loss plot for various functions

# TODO: Add diagram showing architecture of the neural network

def visualize_loss_history(dir_path, model_name, save_dir=None):
    """
    Takes in a path to a directory, returns loss history of all files with model_name
    """
    loss_history = []
    for file in os.listdir(dir_path):
        if file.startswith(model_name):
            model_info = torch.load(file)
            loss = model_info['losses']
            loss_history.extend(loss)
    
    batch = range(len(loss_history))
    plt.plot(batch, loss_history)
    if save_dir:
        plt.savefig(save_dir + "/" + model_name + "loss_history")
    plt.show()

def plot_webtraffic_data(data, opt='first_n', save_dir=None, show=True, n=10):
    "Data is of form [num_timesx[site_name, activity], time]"
    site_info = data[0]
    time = data[1]
    site_names = site_info[:, 0]
    activity = site_info[:, -1:]
    if opt == "first_n":
        for i in range(n):
            plt.plot(time, activity[i, :], label = site_names[i])
        if save_dir:
            plt.savefig(save_dir)
        if show:
            plt.show()



if __name__ == "__main__":
    pass
