import os
import movingpandas as mpd
import matplotlib.pyplot as plt
from datetime import datetime
from common import get_shark_gdf

ids = [
    '2020-04',
    '2020-10',
    '2020-12',
    '2020-13',
    '2020-15',
    '2020-16',
    '2020-17',
    '2020-19',
    '2020-20',
    '2020-21',
    '2020-22',
    '2020-31',
    '2020-32',
    '2020-33',
    '2020-34',
    '2020-35',
    '2020-35_2',
    '2020-36',
    '2020-37',
    '2020-40',
    '2020-41',
    '2020-42'
]

force = False
path = '../data/'
filenames = ['{}-full.csv'.format(id) for id in ids]
savenames = ['{}-full-pos-vs-speed'.format(id) for id in ids]
filepaths = [os.path.join(path, id, filename) for id, filename in zip(ids, filenames)]
savepaths = ['{}{}'.format(os.path.join(path, id, os.path.splitext(filename)[0]), '-pos-vs-speed') for id, filename in zip(ids, filenames)]

lat_lim = (34.3896, 34.4152)
lon_lim = (-119.5277, -119.5716)
lims = [lon_lim, lat_lim]

start_time = datetime.fromisoformat('2020-05-06 07:00:00')
end_time = datetime.fromisoformat('2020-09-04 18:59:00')
norm = 24 * 3600
time_lim = (0, (end_time - start_time).total_seconds() / norm)

boundary_speed = 1.91

for filepath, savepath, id in zip(filepaths, savepaths, ids):
    if not force and os.path.isfile('{}.png'.format(savepath)):
        print('Pos vs speed graph already exists for {}'.format(id))
    else:
        print('Creating pos vs speed graph for {}'.format(id))
        shark_gdf = get_shark_gdf(filepath)
        fig, axs = plt.subplots(1, 3, figsize=(15, 9))
        traj = mpd.TrajectoryCollection(shark_gdf, 'TRANSMITTER').trajectories[0]
        traj.add_speed()
        lower_speeds = traj.df[traj.df['speed'] <= boundary_speed]
        higher_speeds = traj.df[traj.df['speed'] > boundary_speed]
        lower_speed_times = [(time - start_time).total_seconds() / norm for time in lower_speeds['DATETIME']]
        higher_speed_times = [(time - start_time).total_seconds() / norm for time in higher_speeds['DATETIME']]
        axs[2].scatter(lower_speeds['speed'], lower_speed_times, color='black', label='speeds <= {} ({} points)'.format(boundary_speed, len(lower_speeds)))
        axs[2].scatter(higher_speeds['speed'], higher_speed_times, color='red', label='speeds > {} ({} points)'.format(boundary_speed, len(higher_speeds)))
        axs[2].set_xlabel('Speed (m/s)')
        axs[2].set_ylabel('Time in days since 2020-05-06 07:00:00')
        axs[2].legend()
        for ax, col, ylabel, lim in zip(axs, ['LON', 'LAT'], ['Longitude', 'Latitude'], lims):
            ax.scatter(lower_speeds['speed'], lower_speeds[col], color='black', label='speeds <= {} ({} points)'.format(boundary_speed, len(lower_speeds)))
            ax.scatter(higher_speeds['speed'], higher_speeds[col], color='red', label='speeds > {} ({} points)'.format(boundary_speed, len(higher_speeds)))
            ax.set_xlabel('Speed (m/s)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(lim)
            ax.legend()
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()