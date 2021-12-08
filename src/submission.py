import argparse
import os
import pandas as pd
import torch

from data.preprocess import LoadInput, read_data
from models.vae import ODEVAE

def get_submission_data(filename):
    df = pd.read_csv(filename)
    df['date'] = df['page'].str.split('_').str[-1]

    return df

# def generate_predictions(device, model, optimizer, test_loss_func, test_data, test_time, batch_size, n_sample, use_cuda=False):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kaggle Submission Predictions')
    parser.add_argument('--training_dir', type=str, default='data/raw/train_2.csv')
    parser.add_argument('--submission_dir', type=str, default='data/raw/key_2.csv')
    parser.add_argument('--model_save_dir', type=str, default=None, help='Path to where model is stored')
    parser.add_argument('--model_name', type=str, default=None, help='Name of model to load')
    parser.add_argument('--use_cuda', type=eval, default=False)
    args = parser.parse_args()

    if args.training_dir:
        print('Loading training data from file {}...'.format(args.training_dir))
        training_df, training_dates = read_data(args.input)
        if not os.path.exists('../data/processed'):
            os.makedirs('../data/processed')
    else:
        print('No training data found. Exiting...')
        exit()

    if args.submission_dir:
        submission_df = get_submission_data(args.submission_dir)
    else:
        print('No submission data found. Exiting...')
        exit()

    output_dim = 1
    hidden_dim = 64
    latent_dim = 6
    device = 'cpu'
    if args.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Using device: ", device)

    model = ODEVAE(output_dim, hidden_dim, latent_dim).to(device)

    if args.model_save_dir and args.model_name:
        ckpt_path = os.path.join(args.model_save_dir, args.model_name + '.pth')
        print('Loading model from file {}...'.format(ckpt_path))
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No model found. Exiting...')
        exit()

    # predictions_df = generate_predictions()

    