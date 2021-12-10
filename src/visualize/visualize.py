import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from models.vae import mape, differentiable_smape, rounded_smape, kaggle_smape, mae, mse

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

if __name__ == "__main__":
    pass
