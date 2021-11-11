import pandas as pd
import os
from subset import subset
from datetime import datetime

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

input = '../data/SharkArray-01.csv'
# start_time = datetime.fromisoformat('2020-05-06 07:00:00')
start_time = datetime.fromisoformat('2020-08-14 07:00:00')
end_time = datetime.fromisoformat('2020-09-04 18:59:00')
id_filter_mode = 'include'
shark_df = pd.read_csv(input)

for id in ids:
    # output = '../data/{}/{}-full.csv'.format(id, id)
    output = '../data/{}/{}-last-20.csv'.format(id, id)
    if not os.path.isfile(output):
        # print('Creating full subset for {}'.format(id))
        print('Creating last 20 day subset for {}'.format(id))
        subset(input, output, start_time, end_time, [id], id_filter_mode, shark_df=shark_df)
    else:
        # print('Full subset for {} already exists'.format(id))
        print('Last 20 day subset for {} already exists'.format(id))