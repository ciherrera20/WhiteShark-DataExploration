# load modules
import numpy as np
import math
import pandas as pd
import geopandas as gpd
from Shark import *
from matplotlib import pyplot as plt

import matplotlib
matplotlib.style.use('ggplot')


def make_df(filename):
    """ create DataFrame from white shark data,
        it will accept a filename for the CSV data file.

        The data file should be within the same directory
        or the full path must be included in "filename"
    """
    # Read in the datafile
    shark_df = pd.read_csv(filename,
                           #index_col='TRANSMITTER',
                           parse_dates=['DATETIME'])

    # Keep columns of interest only
    shark_df = shark_df[["TRANSMITTER", "DATETIME", "X", "Y", "LAT", "LON", "n", "HPE"]]

    return shark_df


def shark_meta(filename):
    """ Create dataframe of shark metadata

    :param filename: path to data file
    :return: DataFrame of shark tag metadata
    """
    meta_df = pd.read_csv(filename)
    return meta_df


def match_shark_id(shark_ids, meta_df):
    """ Function to extract shark metadata for the sharks in a given position dataset

    :param shark_ids: Series of all shark transmitter IDs
    :param meta_df: dataframe of shark tag metadata

    :return: meta_uniq: same as meta_df but filtered by unique values in tag positions
    """
    # Extract unique transmitter IDs
    uniq_ids = shark_ids.unique()

    # Filter shark metadata by unique IDs
    meta_uniq = meta_df[meta_df.shark.isin(uniq_ids)]

    # Reset index to numerical values (previous index saved as new column in dataframe)
    meta_uniq = meta_uniq.reset_index()

    return meta_uniq


def populate_sharks(shark_data, meta_uniq):
    """ Generate a list of Shark objects with attributes from shark metadata
        and associated position data from shark_data

    :param shark_data: position dataframe
    :param meta_uniq: shark metadata for sharks with position data

    :return: list of Shark objects
    """
    sharks = []
    for index in range(len(meta_uniq)):
        shark_id, size, sex = (meta_uniq.iloc[index].shark,
                               meta_uniq['total natural'][index],
                               meta_uniq['sex'][index])

        # Create a Shark object
        curr_shark = Shark(shark_id, size, sex)

        # Add the shark's data as a DataFrame
        curr_shark.data = shark_data.copy()[shark_data.TRANSMITTER == shark_id]

        # Drop repeated values based on the DATETIME column
        curr_shark.data = curr_shark.data.drop_duplicates(subset=['DATETIME'])

        # Sort time series data by DATETIME column
        curr_shark.data = curr_shark.data.sort_values("DATETIME")

        # Duplicate DATETIME column for time-dependent operations
        curr_shark.data["time"] = curr_shark.data["DATETIME"]

        # Rename the TRANSMITTER column
        curr_shark.data.rename(columns={"TRANSMITTER": "sharkID"}, inplace=True)

        #  Set sharkID as new indices
        # curr_shark.data.set_index(["sharkID"], inplace=True)
        # curr_shark.data.set_index(["sharkID", "time"], inplace=True) #  Set time and sharkID as new indices

        # Set time as new index
        curr_shark.data.set_index(["time"], inplace=True)

        # Append current shark to list
        sharks.append(curr_shark)

    return sharks


def time_between_obs(fish):
    """ compute time difference between consecutive position points

    :param fish: Shark object, all data stored in fish.data (DataFrame)
    :return: delta_t in minutes for current shark
    """

    # Extract DATETIME column
    time = fish.data.DATETIME

    # Sort DATETIME Series, then compute first difference
    delta_t = time.sort_values().diff()

    # Get total seconds, convert to minutes
    delta_t = delta_t.dt.total_seconds().div(60.0).round()

    return delta_t


def plot_delta_t(delta_df, cutoff=720):
    """ plot_detla_t generates histograms and boxplots
        of the time between consecutive observations

        :param delta_df: DataFrame with delta_t values for each shark
        :param cutoff: cutoff value for greatest time difference (minutes) for histograms
    """

    # Frequency histogram, all sharks, color coded
    fig1, axs1 = plt.subplots(figsize=(10, 8))    # Create an empty matplotlib Figure and Axes
    # plt.figure()
    delta_df[delta_df < cutoff].dropna().plot.hist(ax=axs1, alpha=0.4, bins=60)
    axs1.set_xlabel("$\Delta T$ (min)")
    fig1.suptitle('Time between observations', fontsize=12)
    plt.show()

    # Frequency histogram for each shark
    fig2, axs2 = plt.subplots(figsize=(10, 8))
    delta_df[delta_df < cutoff].dropna().hist(ax=axs2, alpha=0.6)
    # delta_df[delta_df < 720].dropna().hist(ax=axs2, alpha=0.6); for array01
    fig2.suptitle('Time between observations by Shark', fontsize=12)
    plt.show()

    # Box plot of data (filtered to values < 720 minutes)
    fig3, axs3 = plt.subplots(figsize=(10, 8))
    delta_df[delta_df < 720].dropna().plot.box(ax=axs3)
    axs3.set_xlabel("$\Delta T$ (min)")
    axs3.tick_params(axis='x', rotation=70)
    fig3.suptitle('Time between observations < 720 min', fontsize=12)
    plt.show()

    # Box plot of data (filtered to values < 240 minutes)
    fig4, axs4 = plt.subplots(figsize=(10, 8))
    delta_df[delta_df < 240].dropna().plot.box(ax=axs4)
    axs4.set_xlabel("$\Delta T$ (min)")
    axs4.tick_params(axis='x', rotation=70)
    fig4.suptitle('Time between observations < 240 min', fontsize=12)
    plt.show()

    # Box plot of data (filtered to values < 30 minutes)
    fig5, axs5 = plt.subplots(figsize=(10, 8))
    delta_df[delta_df < 30].dropna().plot.box(ax=axs5)
    axs5.set_xlabel("$\Delta T$ (min)")
    axs5.tick_params(axis='x', rotation=70)
    fig5.suptitle('Time between observations < 30 min', fontsize=12)
    plt.show()


def calc_step_length(fish):
    """ Compute the distance between consecutive observations

    :param: fish: Shark object with shark information and raw observations DataFrame
    :return: step_length in meters (Series)
    """

    # Extract X-Y position
    x_pos = fish.data.X
    y_pos = fish.data.Y

    # Compute step length between consecutive observations
    step_length = ((x_pos.diff() ** 2) + (y_pos.diff() ** 2)) ** (1/2)

    # Return step length
    return step_length

# TODO: Check units of X,Y data from VPS


def calc_heading_angle(fish):
    """ Compute the turning angle between consecutive observations

    :param: fish: Shark object with shark information and raw observations DataFrame
    :return: heading_angle in radians (Series)
    """

    # Extract X-Y position
    x_pos = fish.data.X
    y_pos = fish.data.Y

    # Compute heading angle between consecutive observations
    heading_angle = np.arctan2(y_pos.diff(), x_pos.diff())

    # Set first value to 0
    heading_angle[0] = 0

    # Return turn angle
    return heading_angle


def calc_turn_angle(fish):
    """ Compute the turning angle between consecutive observations

    :param: fish: Shark object with shark information and raw observations DataFrame
    :return: turn_angle in radians (Series)
    """

    # Extract heading angle
    theta = fish.data.heading_angle

    # Compute turning angle
    turn_angle = np.arctan2(np.sin(theta.diff(periods=-1)), np.cos(theta.diff(periods=-1)))

    # Set first value to 0
    turn_angle[0] = 0

    # Return turn angle
    return turn_angle


def calc_speed(fish):
    """ Compute speed of shark as: step_length / delta_t

    :param: fish: Shark object with shark information and raw observations DataFrame
    :return: speed in meters/sec (Series)
    """

    try:
        speed = fish.data.step_length / (fish.data.delta_t * 60.0)
    except ValueError:
        print("Step length and delta_T are required to compute speed.")

    # Return speed
    return speed


if __name__ == '__main__':
    # Choose what to run by default

    # Run main program with small subset of dataset
    shark_df = make_df('data/subset-calc-pos.csv')

    # Show the first few rows
    shark_df.head()

    # Some print statements to get to know the imported data
    print('The dataset contains', shark_df.shape[0], 'rows and', shark_df.shape[1], 'columns.')
    print('The column names are:', list(shark_df.columns.values))