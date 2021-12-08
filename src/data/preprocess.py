import os
import pandas as pd
import numpy as np
import torch

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
            val_data.append(data[int(len(data) * train_size):int(len(data) * (train_size + val_size))])
            test_data.append(data[int(len(data) * (train_size + val_size)):])
        return train_data, val_data, test_data

    def load_zero_interpolation(self, train_data, val_data, test_data):
        """
        Load the data into the model reshapes it into [page_views, batch_size, features] and
        replaces nan values with 0
        """
        train_data = np.reshape(train_data, (len(train_data), len(train_data[0]), 1))
        train_data = torch.tensor(train_data)
        train_data = train_data.permute(1, 0, 2)
        val_data = np.reshape(val_data, (len(val_data), len(val_data[0]), 1))
        val_data = torch.tensor(val_data)
        val_data = val_data.permute(1, 0, 2)
        test_data = np.reshape(test_data, (len(test_data), len(test_data[0]), 1))
        test_data = torch.tensor(test_data)
        test_data = test_data.permute(1, 0, 2)
        torch.nan_to_num(train_data)
        torch.nan_to_num(val_data)
        torch.nan_to_num(test_data)

        return train_data, val_data, test_data

    def load_average_interpolation(self, train_data, val_data, test_data):
        """
        Load the data into the model reshapes it into [page_views, batch_size, features] and\
        replaces nan values with the average value
        """
        train_data = np.array(train_data)
        train_data_mean = np.nanmean(train_data, axis=1)
        train_idx = np.where(np.isnan(train_data))
        train_data[train_idx] = np.take(train_data_mean, train_idx[1])
        train_data = np.reshape(train_data, (len(train_data), len(train_data[0]), 1))
        train_data = torch.tensor(train_data)
        train_data = train_data.permute(1, 0, 2)

        val_data = np.array(val_data)
        val_data_mean = np.nanmean(val_data, axis=1)
        val_idx = np.where(np.isnan(val_data))
        val_data[val_idx] = np.take(val_data_mean, val_idx[1])
        val_data = np.reshape(val_data, (len(val_data), len(val_data[0]), 1))
        val_data = torch.tensor(val_data)
        val_data = val_data.permute(1, 0, 2)

        test_data = np.array(test_data)
        test_data_mean = np.nanmean(test_data, axis=1)
        test_idx = np.where(np.isnan(test_data))
        test_data[test_idx] = np.take(test_data_mean, test_idx[1])
        test_data = np.reshape(test_data, (len(test_data), len(test_data[0]), 1))
        test_data = torch.tensor(test_data)
        test_data = test_data.permute(1, 0, 2)
        
        return train_data, val_data, test_data
        
    def load_labels(self, train_data, val_data, test_data):
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
    
    def load_time(self, train_data, val_data, test_data):
        """
        Load the time data in the shape of [timesteps, batch_size, features]
        """
        train_time = torch.empty(train_data.shape)
        val_time = torch.empty(val_data.shape)
        test_time = torch.empty(test_data.shape)

        train_data_len = len(train_data)
        val_data_len = len(val_data)
        test_data_len = len(test_data)

        for i in range(train_data_len):
            train_time[i] = torch.full((train_data.shape[1], train_data.shape[2]), i)

        for i in range(train_data_len, train_data_len + val_data_len):
            val_time[i - train_data_len] = torch.full((val_data.shape[1], val_data.shape[2]), i)

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

def gen_batch(x, t, start_row, batch_size, n_sample=100):
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
    
    num_entries = x.shape[1]
    if start_row + batch_size >= num_entries:
        return x[t0_idx:tM_idx, start_row:], t[t0_idx:tM_idx, start_row:]
    return x[t0_idx:tM_idx, start_row:start_row + batch_size], t[t0_idx:tM_idx, start_row: start_row + batch_size] 

# if __name__ == '__main__':
#     train_data, val_data, test_data, train_time, val_time, test_time = run()
