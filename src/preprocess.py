import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse

def read_data(filename):
    """
    Read data from csv file
    """
    df = pd.read_csv(filename)
    dates = [i for i in df.columns if i != 'Page']

    return df, dates

def parse_url(df):
    """
    Parse url from Page column
    """
    page = df['Page'].str.split('_')
    df['name'] = page.str[0]
    df['project'] = page.str[1]
    df['access'] = page.str[2]
    df['agent'] = page.str[3]

    return df

def label_encode(df):
    """
    Label encode columns
    """
    le = LabelEncoder()
    df['page_url'] = le.fit_transform(df['Page'])
    df['project'] = le.fit_transform(df['project'])
    df['access'] = le.fit_transform(df['access'])
    df['agent'] = le.fit_transform(df['agent'])

    return df

def run():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--input', type=str, help='Input file')
    args = parser.parse_args()

    df, dates = read_data(args.input)
    df = parse_url(df)
    df = label_encode(df)
    train_page_views = df[dates].values
    
    if not os.path.exists('data/parsed'):
        os.makedirs('data/parsed')

    # Saves a map of the label to the value
    df[['page_url', 'Page']].to_csv('data/parsed/ids_to_page.csv', index=False)
    # Saves data
    np.save('data/parsed/train_page_views.npy', train_page_views)
    np.save('data/parsed/project_names.npy', df['project'].values)
    np.save('data/parsed/access_names.npy', df['access'].values)
    np.save('data/parsed/agent_names.npy', df['agent'].values)
    np.save('data/parsed/page_names.npy', df['page_url'].values)

if __name__ == '__main__':
    run()
