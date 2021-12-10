import os
import pandas as pd
import numpy as np
import torch

np.set_printoptions(threshold=500)

class LoadInput:
    def __init__(self, data_path):
        file_path = [
            '/page_views.npy',
        ]
        self.file_path = data_path + file_path[0]
        self.loaded_data = np.load(self.file_path)

    def split_train_val_test(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Split the loaded data into train, validation and test data
        """
        train_data = []
        val_data = []
        test_data = []
        for data in self.loaded_data:
            train_data.append(data[:int(len(data) * train_size)])
            if val_size != 0:
                val_data.append(data[int(len(data) * train_size):int(len(data) * (train_size + val_size))])
            if test_size != 0:
                test_data.append(data[int(len(data) * (train_size + val_size)):])
        return train_data, val_data, test_data
    
def remove_percent_nan_values(data, percent):
    percent_nan_per_row = np.isnan(data).sum(1)/data.shape[1]
    data = data[percent_nan_per_row < percent]

    return data

def load_zero_interpolation(train_data, val_data, test_data):
    """
    Load the data into the model reshapes it into [page_views, batch_size, features] and
    replaces nan values with 0
    """
    if train_data is not None:
        train_data = np.array(train_data)
        train_data = np.expand_dims(train_data, axis=2)
        train_data = torch.tensor(train_data)
        train_data = train_data.permute(1, 0, 2)
        train_data = torch.nan_to_num(train_data)

    if val_data is not None:
        val_data = np.array(val_data)
        val_data = np.expand_dims(val_data, axis=2)
        val_data = torch.tensor(val_data)
        val_data = val_data.permute(1, 0, 2)
        val_data = torch.nan_to_num(val_data)

    if test_data is not None:
        test_data = np.array(test_data)
        test_data = np.expand_dims(test_data, axis=2)
        test_data = torch.tensor(test_data)
        test_data = test_data.permute(1, 0, 2)
        test_data = torch.nan_to_num(test_data)

    return train_data, val_data, test_data

def load_average_interpolation(train_data, val_data, test_data):
    """
    Load the data into the model reshapes it into [page_views, batch_size, features] and\
    replaces nan values with the average value
    """
    if train_data is not None:
        train_data = np.array(train_data)
        train_data = remove_percent_nan_values(train_data, 0.1)
        train_data = np.where(np.isnan(train_data), np.ma.array(train_data, mask=np.isnan(train_data)).mean(axis=1)[:, np.newaxis], train_data)
        train_data = np.expand_dims(train_data, axis=2)
        train_data = torch.tensor(train_data)
        train_data = train_data.permute(1, 0, 2)

    if val_data is not None:
        val_data = np.array(val_data)
        val_data = remove_percent_nan_values(val_data, 0.1)
        val_data = np.where(np.isnan(val_data), np.ma.array(val_data, mask=np.isnan(val_data)).mean(axis=1)[:, np.newaxis], val_data)
        val_data = np.expand_dims(val_data, axis=2)
        val_data = torch.tensor(val_data)
        val_data = val_data.permute(1, 0, 2)

    if test_data is not None:
        test_data = np.array(test_data)
        test_data = remove_percent_nan_values(test_data, 0.1)
        test_data = np.where(np.isnan(test_data), np.ma.array(test_data, mask=np.isnan(test_data)).mean(axis=1)[:, np.newaxis], test_data)
        test_data = np.expand_dims(test_data, axis=2)
        test_data = torch.tensor(test_data)
        test_data = test_data.permute(1, 0, 2)
    
    return train_data, val_data, test_data

# def load_average_interpolation_with_noise(train_data, val_data, test_data, noise_level=0.08):
#     """
#     Load the data into the model reshapes it into [page_views, batch_size, features] and\
#     replaces nan values with the average value
#     """
#     if train_data is not None:
#         train_data = np.array(train_data)
#         # train_data = remove_percent_nan_values(train_data, 0.1)
#         mean_mask = np.ma.array(train_data, mask=np.isnan(train_data)).mean(axis=1)
#         train_data = np.where(np.isnan(train_data), np.random.randint((1 - noise_level) * mean_mask[:, np.newaxis], (1 + noise_level) * mean_mask[:, np.newaxis]), train_data)
#         train_data = np.expand_dims(train_data, axis=2)
#         train_data = torch.tensor(train_data)
#         train_data = train_data.permute(1, 0, 2)

#     if val_data is not None:
#         val_data = np.array(val_data)
#         val_data = remove_percent_nan_values(val_data, 0.1)
#         mean_mask = np.ma.array(val_data, mask=np.isnan(val_data)).mean(axis=1)
#         val_data = np.where(np.isnan(val_data), np.random.randint((1 - noise_level) * mean_mask[:, np.newaxis], (1 + noise_level) * mean_mask[:, np.newaxis]), val_data)
#         val_data = np.expand_dims(val_data, axis=2)
#         val_data = torch.tensor(val_data)
#         val_data = val_data.permute(1, 0, 2)

#     if test_data is not None:
#         test_data = np.array(test_data)
#         test_data = remove_percent_nan_values(test_data, 0.1)
#         mean_mask = np.ma.array(test_data, mask=np.isnan(test_data)).mean(axis=1)
#         test_data = np.where(np.isnan(test_data), np.random.randint((1 - noise_level) * mean_mask[:, np.newaxis], (1 + noise_level) * mean_mask[:, np.newaxis]), test_data)
#         test_data = np.expand_dims(test_data, axis=2)
#         test_data = torch.tensor(test_data)
#         test_data = test_data.permute(1, 0, 2)
    
#     return train_data, val_data, test_data

def load_median_interpolation(train_data, val_data, test_data):
    """
    Load the data into the model reshapes it into [page_views, batch_size, features] and\
    replaces nan values with the average value
    """
    if train_data is not None:
        train_data = np.array(train_data)
        train_data = remove_percent_nan_values(train_data, 0.1)
        median = np.ma.median(np.ma.array(train_data, mask=np.isnan(train_data)), axis=1)
        train_data = np.where(np.isnan(train_data), median[:, np.newaxis], train_data) 
        train_data = np.expand_dims(train_data, axis=2)
        train_data = torch.tensor(train_data)
        train_data = train_data.permute(1, 0, 2)

    if val_data is not None:
        val_data = np.array(val_data)
        val_data = remove_percent_nan_values(val_data, 0.1)
        median = np.ma.median(np.ma.array(val_data, mask=np.isnan(val_data)), axis=1)
        val_data = np.where(np.isnan(val_data), median[:, np.newaxis], val_data) 
        val_data = np.expand_dims(val_data, axis=2)
        val_data = torch.tensor(val_data)
        val_data = val_data.permute(1, 0, 2)

    if test_data is not None:
        test_data = np.array(test_data)
        test_data = remove_percent_nan_values(test_data, 0.1)
        median = np.ma.median(np.ma.array(test_data, mask=np.isnan(test_data)), axis=1)
        test_data = np.where(np.isnan(test_data), median[:, np.newaxis], test_data)
        test_data = np.expand_dims(test_data, axis=2)
        test_data = torch.tensor(test_data)
        test_data = test_data.permute(1, 0, 2)
    
    return train_data, val_data, test_data
    
# def load_median_interpolation_with_noise(train_data, val_data, test_data, noise_level=0.08):
#     if train_data is not None:
#         train_data = np.array(train_data)
#         # train_data = remove_percent_nan_values(train_data, 0.1)
#         median = np.ma.median(np.ma.array(train_data, mask=np.isnan(train_data)), axis=1)
#         train_data = np.where(np.isnan(train_data), np.random.randint((1 - noise_level) * median[:, np.newaxis], (1 + noise_level) * median[:, np.newaxis]) , train_data) 
#         train_data = np.expand_dims(train_data, axis=2)
#         train_data = torch.tensor(train_data)
#         train_data = train_data.permute(1, 0, 2)

#     if val_data is not None:
#         val_data = np.array(val_data)
#         val_data = remove_percent_nan_values(val_data, 0.1)
#         median = np.ma.median(np.ma.array(val_data, mask=np.isnan(val_data)), axis=1)
#         val_data = np.where(np.isnan(val_data), np.random.randint((1 - noise_level) * median[:, np.newaxis], (1 + noise_level) * median[:, np.newaxis]), val_data) 
#         val_data = np.expand_dims(val_data, axis=2)
#         val_data = torch.tensor(val_data)
#         val_data = val_data.permute(1, 0, 2)

#     if test_data is not None:
#         test_data = np.array(test_data)
#         test_data = remove_percent_nan_values(test_data, 0.1)
#         median = np.ma.median(np.ma.array(test_data, mask=np.isnan(test_data)), axis=1)
#         test_data = np.where(np.isnan(test_data), np.random.randint((1 - noise_level) * median[:, np.newaxis], (1 + noise_level) * median[:, np.newaxis]), test_data)
#         test_data = np.expand_dims(test_data, axis=2)
#         test_data = torch.tensor(test_data)
#         test_data = test_data.permute(1, 0, 2)
    
#     return train_data, val_data, test_data

def load_labels(train_data, val_data, test_data):
    """
    Labels are whether page views are increasing or decreasing of size [batch_size, timesteps]
    """
    train_labels = []
    val_labels = []
    test_labels = []
    for data in train_data:
        train_labels.append(np.array([1 if data[i+1] > data[i] else 0 for i in range(len(data)-1)]))
    for data in val_data:
        val_labels.append(np.array([1 if data[i+1] > data[i] else 0 for i in range(len(data)-1)]))
    for data in test_data:
        test_labels.append(np.array([1 if data[i+1] > data[i] else 0 for i in range(len(data)-1)]))
    return train_labels, val_labels, test_data

def load_time(train_data, val_data, test_data):
    """
    Load the time data in the shape of [timesteps, batch_size, features]
    """
    train_time = None
    val_time = None
    test_time = None

    if train_data is not None:
        train_time = torch.empty(train_data.shape)
        train_data_len = len(train_data)
        for i in range(train_data_len):
            train_time[i] = torch.full((train_data.shape[1], train_data.shape[2]), i)
    
    if val_data is not None:
        val_time = torch.empty(val_data.shape)
        val_data_len = len(val_data)
        for i in range(train_data_len, train_data_len + val_data_len):
            val_time[i - train_data_len] = torch.full((val_data.shape[1], val_data.shape[2]), i)

    if test_data is not None:
        test_time = torch.empty(test_data.shape)
        test_data_len = len(test_data)
        for i in range(train_data_len + val_data_len, train_data_len + val_data_len + test_data_len):
            test_time[i - train_data_len - val_data_len] = torch.full((test_data.shape[1], test_data.shape[2]), i)

    return train_time, val_time, test_time

def read_data(filename):
    """
    Read data from csv file
    """
    df = pd.read_csv(filename)
    dates = [i for i in df.columns if i != 'Page']

    return df, dates

def gen_batch(x, t, batch_indices, n_sample=100):
    """
    Generate batches of data
    Input: x: Data of size (num_timsteps, num_rows, 1)
           t: Timesteps of size (num_timesteps, num_rows, 1)
           batch_size: desired batch size
           n_sample: number of timesteps to sample
    Output: Data of size (n_sample, batch_size, 1)
            Timesteps of size (n_sample, batch_size, 1)
    """
    time_len = x.shape[0]
    n_sample = min(n_sample, time_len)
    if n_sample > 0:
        t0_idx = np.random.multinomial(1, [1. / (time_len - n_sample)] * (time_len - n_sample))
        t0_idx = np.argmax(t0_idx)
        tM_idx = t0_idx + n_sample
    else:
        t0_idx = 0
        tM_idx = time_len
    
    return x[t0_idx:tM_idx][:, batch_indices], t[t0_idx:tM_idx][:, batch_indices] 

def get_rows(x, start_col, num_cols, start_time, time_len):
    # End time is exclusive
    end_time = min(start_time + time_len, x.shape[0])
    return x[start_time:end_time][:, start_col:start_col + num_cols]


# if __name__ == '__main__':

