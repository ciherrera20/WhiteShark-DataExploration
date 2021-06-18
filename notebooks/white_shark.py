# load modules
import numpy as np
import pandas as pd
import math

def make_df(filename):
    """ main will run a simple analysis of white shark data,
        it will accept a filename for the CSV data file
    """
    shark_df = pd.read_csv(filename,
                           #index_col='TRANSMITTER',
                           parse_dates=['DATETIME'])
    return shark_df

if __name__ == '__main__':
    # Choose what to run by default

    # Run main program with small subset of dataset
    shark_df = make_df('subset-calc-pos.csv')

    # Show the first few rows
    shark_df.head()

    # Some print statements to get to know the imported data
    print('The dataset contains', shark_df.shape[0], 'rows and', shark_df.shape[1], 'columns.')
    print('The column names are:', list(shark_df.columns.values))