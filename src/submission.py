import argparse
import os
import pandas as pd
import numpy as np
import torch
import math
import time

from data.preprocess import LoadInput, read_data, get_rows, load_median_interpolation
from models.vae import ODEVAE

def submission_data_to_df(file_path, start_row, num_rows=100000):
    start_row = max(1, start_row)
    df = pd.read_csv(file_path, header=0, skiprows=start_row, nrows=num_rows)
    return df

def get_submission_timesteps(df, start_date="2015-07-01"):
    def map_page_to_time(date, start_time):
        date_split = date.split("-")
        time = int(date_split[0]) * 365 + days_up_to[int(date_split[1])] + int(date_split[2])
        return time - start_time
    
    # df = pd.read_csv(filename)

    days_up_to = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151, 7: 181, 8: 212, 9: 243, 10: 273, 11: 304, 12: 334}

    start_date_split = str.split(start_date, "-")
    start_time = int(start_date_split[0]) * 365 + days_up_to[int(start_date_split[1])] + int(start_date_split[2])
    
    df['Time'] = df['Page'].map(lambda p: p.rsplit('_', 1)[1])
    df['Page'] = df['Page'].map(lambda p: p.rsplit('_', 1)[0])
    df['Time'] = df['Time'].map(lambda d: map_page_to_time(d, start_time))
    return df

def generate_predictions(device, model, training_df, predict_date_range, n_sample, batch_size=1000, use_cuda=False):
    result = {}
    training_data = [cn for cn in training_df.columns if cn != 'Page']
    training_data = training_df[training_data].values
    training_data = training_data[:, training_data.shape[1] - n_sample:]
    training_data, _, _ = load_median_interpolation(training_data, None, None)
        
    training_times = torch.arange(0, training_data.shape[0], dtype=torch.float32)

    num_batches = math.ceil(training_data.shape[1]/batch_size)
    print("Num batches: {}\n".format(num_batches))
    
    for i in range(num_batches):
        start_time = time.time()
        
        if (i * batch_size) + batch_size > training_data.shape[1]:
            batch_size = training_data.shape[1]
        
        batch_x = get_rows(training_data, (i * batch_size), batch_size, len(training_data) - n_sample, n_sample) 
        batch_t_encoder = training_times[len(training_data) - n_sample:]
        batch_t_encoder = batch_t_encoder.repeat(batch_x.shape[1], 1).permute(1, 0).unsqueeze(2)

        batch_t_decoder = torch.arange(predict_date_range[0], predict_date_range[1], 1).float()
        batch_t_decoder = batch_t_decoder.repeat(batch_x.shape[1], 1).permute(1, 0).unsqueeze(2)

        print("Batch x shape: ", batch_x.shape)
        print("Batch t encoder shape: ", batch_t_encoder.shape)
        print("Batch t decoder shape: ", batch_t_decoder.shape)

        x_p, _, _, _ = model(batch_x, batch_t_encoder, batch_t_decoder)
        x_p = x_p.squeeze(2).permute(1, 0)

        print("x_p shape: ", x_p.shape)
        
        for j in range(i * batch_size, i * batch_size + batch_size):
            result[training_df['Page'][j]] = x_p[j - (i * batch_size + batch_size)].detach().numpy()

        end_time = time.time()
        time_taken = end_time - start_time

        print("Time taken: {}s".format(round(time_taken, 3)))
        # submission_df = submission_data_to_df(test_file_path, idx * batch_size)
        # submission_df = get_submission_timesteps(submission_df)
        # submission_df = submission_df.merge(training_df, on='Page', how='left')

    return result

def main():
    parser = argparse.ArgumentParser(description='Kaggle Submission Predictions')
    parser.add_argument('--training_dir', type=str, default='../data/raw/train_2.csv')
    parser.add_argument('--submission_dir', type=str, default='../data/raw/key_2.csv')
    parser.add_argument('--model_save_dir', type=str, default=None, help='Path to where model is stored')
    parser.add_argument('--model_name', type=str, default=None, help='Name of model to load')
    parser.add_argument('--use_cuda', type=eval, default=False)
    args = parser.parse_args()

    if args.training_dir:
        print('Loading training data from file {}...'.format(args.training_dir))
        training_df, training_dates = read_data(args.training_dir)
        if not os.path.exists('../data/processed'):
            os.makedirs('../data/processed')
        
        
        # ld = LoadInput('../data/processed')
        # train_data, val_data, test_data = ld.split_train_val_test(train_size=1, val_size=0, test_size=0)

        # train_data, val_data, test_data = ld.load_average_interpolation(train_data, val_data, test_data)
        # train_time, val_time, test_time = ld.load_time(train_data, val_data, test_data)
    else:
        print('No training data found. Exiting...')
        exit()

    if args.submission_dir:
        submission_df = pd.read_csv(args.submission_dir)
        submission_df = get_submission_timesteps(submission_df)
        # End index for date range is exclusive
        date_range = [submission_df['Time'].min(), submission_df['Time'].max() + 1]
        training_df = training_df[training_df['Page'].isin(submission_df['Page'].unique())]
    else:
        print('No submission data found. Exiting...')
        exit()

    output_dim = 1
    hidden_dim = 64
    latent_dim = 6
    n_sample = 200
    batch_size = 1000
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

    # test_time = torch.tensor(submission_df['Time'].values).to(device)
    predictions = generate_predictions(device, model, training_df, date_range, n_sample, batch_size)
    print(predictions.shape)


if __name__ == "__main__":
    main()

    