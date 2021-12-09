import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from data.preprocess import mape, differentiable_smape, rounded_smape, kaggle_smape, mae, mse

# sns.set_theme(style="darkgrid")

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

if __name__ == "__main__":
    pass
