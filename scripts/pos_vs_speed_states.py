import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import get_shark_gdf
import re
from datetime import datetime
import numpy as np

force = False
pattern = re.compile(r'^(\d)s-(?=[tsd])(t?s?d?)-states.csv$')
data_dir = '../graphs'
paths = []
states = []
obs = []
for dir in os.listdir(data_dir):
    abs_dir = os.path.join(data_dir, dir)
    if os.path.isdir(abs_dir):
        for filename in os.listdir(abs_dir):
            m = pattern.match(filename)
            if m is not None:
                g = m.groups()
                paths.append(abs_dir)
                states.append(int(g[0]))
                obs.append(g[1])

filename = '{}s-{}-states.csv'
filepaths = [os.path.join(path, filename.format(state, ob)) for path, state, ob in zip(paths, states, obs)]
savepaths = [os.path.join(path, '{}s-{}-pos-vs-speed'.format(state, ob)) for path, state, ob in zip(paths, states, obs)]

lat_lim = (34.3896, 34.4152)
lon_lim = (-119.5277, -119.5716)
lims = [lon_lim, lat_lim, None]

start_time = datetime.fromisoformat('2020-05-06 07:00:00')
end_time = datetime.fromisoformat('2020-09-04 18:59:00')
norm = 24 * 3600
time_lim = (0, (end_time - start_time).total_seconds() / norm)

for filepath, savepath, path, num_states in zip(filepaths, savepaths, paths, states):
    if not force and os.path.isfile('{}.png'.format(savepath)):
        print('Pos vs speed graph already exists for {}'.format(os.path.normpath(savepath)))
    else:
        print('Creating pos vs speed graph for {}'.format(os.path.normpath(savepath)))
        shark_gdf = get_shark_gdf(filepath)
        shark_gdf['days'] = [(time - start_time).total_seconds() / norm for time in shark_gdf['DATETIME']]
        state_gdfs = []
        for state in range(num_states):
            state_gdfs.append(shark_gdf[shark_gdf['state'] == state])

        fig, axs = plt.subplots(1, 3, figsize=(15, 9))
        cmaplist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:num_states]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, num_states)

        for ax, col, ylabel, lim in zip(axs, ['LON', 'LAT', 'days'], ['Longitude', 'Latitude', 'Time in days since 2020-05-06 07:00:00'], lims):
            # state_gdf = shark_gdf[shark_gdf['state'] == state]
            for state_gdf, inner_state in zip(state_gdfs, range(num_states)):
                ax.scatter(state_gdf['speed'], state_gdf[col], label='state {} ({} points)'.format(inner_state, len(state_gdf)))
            for i, (state_gdf, inner_state) in enumerate(zip(state_gdfs, range(num_states))):
                mean_speed = np.average(np.array(state_gdf['speed']))
                mean_pos = np.average(np.array(state_gdf[col]))
                color = cmap(i)
                ax.scatter([mean_speed], [mean_pos], color=color, cmap=cmap, marker='o', s=70, edgecolors='black', label='state {}, mean ({:.3f}, {:.3f})'.format(inner_state, mean_speed, mean_pos))
            ax.set_xlabel('Speed (m/s)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(lim)
            ax.legend()
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()