import pandas as pd
import movingpandas as mpd
import matplotlib.pyplot as plt
from datetime import datetime
from common import get_shark_gdf

# filepath = '../data/2020-35_2/2020-35_2-full.csv'
filepath = '../data/SharkArray-01.csv'
savepath = '../data/SharkArray-01-pos-vs-speed'

shark_gdf = get_shark_gdf(filepath)
trajs = mpd.TrajectoryCollection(shark_gdf, 'TRANSMITTER')
for traj in trajs:
    traj.add_speed()

# print(full_df.columns, full_df.dtypes)
# print(full_df['DATETIME'][0], type(full_df['DATETIME'][0]))

lat_lim = (34.3896, 34.4152)
lon_lim = (-119.5277, -119.5716)
lims = [lon_lim, lat_lim]

start_time = datetime.fromisoformat('2020-05-06 07:00:00')
end_time = datetime.fromisoformat('2020-09-04 18:59:00')
norm = 24 * 3600
time_lim = (0, (end_time - start_time).total_seconds() / norm)

fig, axs = plt.subplots(1, 3, figsize=(3 * 5, 9))
full_df = pd.concat([traj.df for traj in trajs])

boundary_speed = 1.91

lower_speeds = full_df[full_df['speed'] <= boundary_speed]
higher_speeds = full_df[full_df['speed'] > boundary_speed]
lower_speed_times = [(time - start_time).total_seconds() / norm for time in lower_speeds['DATETIME']]
higher_speed_times = [(time - start_time).total_seconds() / norm for time in higher_speeds['DATETIME']]
axs[2].scatter(lower_speeds['speed'], lower_speed_times, color='black', label='speeds <= {} ({} points)'.format(boundary_speed, len(lower_speeds)))
axs[2].scatter(higher_speeds['speed'], higher_speed_times, color='red', label='speeds > {} ({} points)'.format(boundary_speed, len(higher_speeds)))
axs[2].set_xlabel('Speed (m/s)')
axs[2].set_ylabel('Time in days since 2020-05-06 07:00:00')
axs[2].legend()
for ax, col, ylabel, lim in zip(axs[0:2], ['LON', 'LAT'], ['Longitude', 'Latitude'], lims):
    ax.scatter(lower_speeds['speed'], lower_speeds[col], color='black', label='speeds <= {} ({} points)'.format(boundary_speed, len(lower_speeds)))
    ax.scatter(higher_speeds['speed'], higher_speeds[col], color='red', label='speeds > {} ({} points)'.format(boundary_speed, len(higher_speeds)))
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(lim)
    ax.legend()
plt.savefig(savepath, bbox_inches='tight')
plt.close()