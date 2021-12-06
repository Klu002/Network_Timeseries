import os
import pandas as pd
import numpy as np
import argparse

class LoadInput:
    def __init__(self, data_path):
        file_path = [
            'page_views.npy',
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
    
    def load_data(self, train_data, val_data, test_data):
        """
        Load the data into the model reshapes it into [page_views, batch_size, features]
        """
        train_data = np.reshape(train_data, (len(train_data[0]), len(train_data), 1))
        val_data = np.reshape(val_data, (len(val_data[0]), len(val_data), 1))
        test_data = np.reshape(test_data, (len(test_data[0]), len(test_data), 1))
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
        train_time = []
        val_time = []
        test_time = []
        for data in train_data:
            train_time.append(np.array([i for i in range(len(data))]))
        for data in val_data:
            val_time.append(np.array([i for i in range(len(data))]))
        for data in test_data:
            test_time.append(np.array([i for i in range(len(data))]))
        
        train_time = np.reshape(train_time, (len(train_time[0]), len(train_time), 1))
        val_time = np.reshape(val_time, (len(val_time[0]), len(val_time), 1))
        test_time = np.reshape(test_time, (len(test_time[0]), len(test_time), 1))

        return train_time, val_time, test_time


def read_data(filename):
    """
    Read data from csv file
    """
    df = pd.read_csv(filename)
    dates = [i for i in df.columns if i != 'Page']

    return df, dates

def run():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--load', type=str, help='Load data from file')
    args = parser.parse_args()
  
    if args.input:
        print('Reading data from file')
        data_path = args.input
        df, dates = read_data(data_path)
        if not os.path.exists('data/parsed'):
            os.makedirs('data/parsed')

        # Saves data
        np.save('data/parsed/page_views.npy', df[dates].values)
    
    if args.load:
        print('Loading data from file')
        data_path = args.load
        ld = LoadInput(data_path)
        train_data, val_data, test_data = ld.split_train_val_test()
        train_data, val_data, test_data = ld.load_data(train_data, val_data, test_data)
        train_time, val_time, test_time = ld.load_time(train_data, val_data, test_data)
        return train_data, val_data, test_data, train_time, val_time, test_time
    

if __name__ == '__main__':
    run()
