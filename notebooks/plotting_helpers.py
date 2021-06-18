import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta
import movingpandas as mpd
import contextily as ctx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from pyproj import CRS
from keplergl import KeplerGl

# Function to get string representing a trajectory's start and end times
def start_to_end(traj):
    return str(traj.get_start_time()) + ' to ' + str(traj.get_end_time())

# Returns whether or not a time falls between a trajectory's start and end time
def traj_contains_time(traj, time):
    return traj.get_start_time() <= time <= traj.get_end_time()

# Return a list of the trajectories in the given trajectory collection that fall within the given start time and end time
def filter_traj_col(traj_col, start_time, end_time):
    trajs = []
    for traj in traj_col.trajectories:
        if start_time <= traj.get_start_time() and traj.get_end_time() <= end_time:
            trajs.append(traj)
    return mpd.TrajectoryCollection(trajs)

# Get xlim and ylim based on the bounding box around a trajectory, padded by some percentage
def get_lims(traj, xlim=None, ylim=None, padding=1):
    if xlim == None or ylim == None:
        # Retrieve trajectory bounds
        minx, miny, maxx, maxy = traj.get_bbox()
        if xlim == None:
            xlim = (minx, maxx)
        if ylim == None:
            ylim = (miny, maxy)

    if padding != 1:
        width = (xlim[1] - xlim[0]) * padding
        height = (ylim[1] - ylim[0]) * padding
        centerx = sum(xlim) / 2
        centery = sum(ylim) / 2
        xlim = (centerx - width / 2, centerx + width / 2)
        ylim = (centery - width / 2, centery + width / 2)
    return xlim, ylim

# Function to plot a collection of trajectories on the same axes
def plot_trajs(traj_col, title=None, legend=False, xlim=None, ylim=None, padding=1, rep_traj=None, figsize=(9, 5), receivers=None):
    if rep_traj == None:
        rep_traj = traj_col.trajectories[0]
    cmap = plt.get_cmap('jet')
    N = len(traj_col.trajectories)
    xlim, ylim = get_lims(rep_traj, xlim=xlim, ylim=ylim, padding=padding)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot and label each trajectory
    for i, traj in enumerate(traj_col.trajectories):
        color = cmap(float(i) / N)
        label = start_to_end(traj)
        traj.plot(ax=ax, linewidth=5, capstyle='round', label=label, color=color)
    
    # Plot receiver positions
    if receivers is not None:
        receivers.plot(ax=ax)
    
    # Set up axes
    if legend:
        ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Download map tiles and use them as a background image
    west, south, east, north = (xlim[0], ylim[0], xlim[1], ylim[1])
    ctx.add_basemap(ax, crs=rep_traj.crs.to_string(), source=ctx.providers.Stamen.Terrain)
    
    plt.title(title)
    plt.show()

# Animate a collection of trajectories
def animate_trajectories(trajectories, num_frames=100, start_time=None, end_time=None, xlim=None, ylim=None, padding=1, interval=20, rep_traj=None, figsize=(9, 9), receivers=None):
    # Populate keyword arguments if necessary
    if rep_traj == None:
        rep_traj = trajectories[0]
    if start_time == None:
        start_time = rep_traj.get_start_time()
    if end_time == None:
        end_time = rep_traj.get_end_time()
    duration = end_time - start_time
    xlim, ylim = get_lims(rep_traj, xlim=xlim, ylim=ylim, padding=padding)
    N = len(trajectories)
    cmap = plt.get_cmap('jet')
    
    lag_time = 500
    lag_frames = lag_time // interval
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    title = ax.text(sum(xlim) / 2, ylim[1] - ((ylim[1] - ylim[0]) / 100), "", horizontalalignment='center', verticalalignment='top')
    positions = []
    paths = []
    paths_data = []
    for i, traj in enumerate(trajectories):
        color = cmap(float(i) / N)
        positions.append(ax.plot([], [], linestyle='None', color=color, marker='o', label=traj.df['TRANSMITTER'][0])[0])
        paths.append(ax.plot([], [], linestyle='-', color=color, marker=',')[0])
        paths_data.append(([], []))
    
    # Set up axes
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Download map tiles and use them as a background image
    west, south, east, north = (xlim[0], ylim[0], xlim[1], ylim[1])
    ctx.add_basemap(ax, crs=rep_traj.crs.to_string(), source=ctx.providers.Stamen.Terrain)
    
    def init():
        # Plot receiver positions
        if receivers is not None:
            receivers.plot(ax=ax)
        return (title, *tuple(positions), *tuple(paths))
    
    def update(frame):
        frame_time = start_time + (duration * frame / num_frames)
        for traj, position, path, path_data in zip(trajectories, positions, paths, paths_data):
            if traj_contains_time(traj, frame_time):
                point = traj.get_position_at(frame_time)
                x, y = point.coords[0]
                path_data_x, path_data_y = path_data
                path_data_x.append(x)
                path_data_y.append(y)
                position.set_data([x], [y])
            else:
                position.set_data([], [])
            path_data_x = path_data_x[-lag_frames:]
            path_data_y = path_data_y[-lag_frames:]
            path.set_data(path_data_x, path_data_y)
        title.set_text(frame_time)
        return (title, *tuple(positions), *tuple(paths))
    
    return FuncAnimation(fig, update, frames=range(0, num_frames), init_func=init, blit=True, interval=interval, repeat=False)