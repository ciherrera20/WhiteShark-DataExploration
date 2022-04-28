import movingpandas as mpd
from common import get_shark_gdf
from datetime import timedelta
import numpy as np
from subset import subset
from common import mkdir
import os

def extract_runs(input, minutes, id):
    print('Getting shark gdf...')
    shark_gdf = get_shark_gdf(input)
    traj = mpd.TrajectoryCollection(shark_gdf, 'TRANSMITTER').trajectories[0]
    print('Splitting shark trajectory by observation gap...')
    all_split_trajs = mpd.ObservationGapSplitter(traj).split(gap=timedelta(minutes=minutes))
    split_trajs = [trajectory for trajectory in all_split_trajs.trajectories if trajectory.size() > 100]
    print('Creating output directory...')
    outputdir = '../data/{}/runs'.format(id)
    mkdir(outputdir)
    print('Saving trajectories...')
    for split_traj in split_trajs:
        output = os.path.join(outputdir, '{}-long-{}-min.csv'.format(split_traj.size(), minutes))
        print(output)
        subset(input, output, split_traj.get_start_time(), split_traj.get_end_time(), [split_traj.id.split('_')[0]], 'include', shark_df=shark_gdf)

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

# ids = [
#     '2020-13'
# ]

minutes = 3
for id in ids:
    print('Extracting runs for {}'.format(id))
    input = '../data/{}/{}-full-temps.csv'.format(id, id)
    # input = '../data/{}/{}-full.csv'.format(id, id)
    try:
        extract_runs(input, minutes, id)
    except:
        print('Unable to extract runs for {}'.format(id))