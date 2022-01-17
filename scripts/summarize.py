import pandas as pd
from datetime import datetime
import argparse
import os

def summarize(input):
    shark_df = pd.read_csv(input)
    shark_df['DATETIME'] = pd.to_datetime(shark_df['DATETIME'])
    print('Columns:\n\t', end='')
    print(shark_df.columns)
    print(shark_df.head())
    print('Length:\n\t', end='')
    print(len(shark_df))
    print('Transmitter IDs:\n\t', end='')
    print(shark_df['TRANSMITTER'].unique())
    print('Start time:\n\t', end='')
    print(shark_df['DATETIME'].min())
    print('End time:\n\t', end='')
    print(shark_df['DATETIME'].max())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print a summary of the given data.')
    parser.add_argument('input', help='CSV file containing the input data')
    args = parser.parse_args()
    summarize(args.input)
