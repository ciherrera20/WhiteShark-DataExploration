import pandas as pd
from datetime import datetime
import argparse
import yaml
import os

def filter_time(df, start_time=datetime(2020, 5, 21, 20, 0, 0), end_time=datetime(2020, 5, 21, 23, 0, 0)):
    '''Filter out rows whose time does not fall between the given start and end times.'''
    dates = pd.to_datetime(df['DATETIME'])
    if start_time == None and end_time == None:
        return df
    elif start_time == None:
        return df[(dates <= end_time)]
    elif end_time == None:
        return df[(start_time <= dates)]
    else:
        return df[(start_time <= dates) & (dates <= end_time)]

def filter_id(df, id_list = ['2020-19'], mode='include'):
    '''Filter out whose transmitter id is in the id list, or is not in the id list if mode is set to exclude.'''
    if mode == 'include':
        df = df[df['TRANSMITTER'].isin(id_list)]
    elif mode == 'exclude':
        df = df[~df['TRANSMITTER'].isin(id_list)]
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a subset of telemetry data by filtering over a time range and over a list of transmitter ids.')
    parser.add_argument('input', nargs='?', help='CSV file containing the input telemetry data.')
    parser.add_argument('output', nargs='?', help='CSV file containing the output telemetry data subset. Defaults to \'./subset.csv\'.')
    parser.add_argument('-c', '--config', help='Name of the config yaml file to use. All of the other arguments can be provided there using the long argument name without the starting dashes.')
    parser.add_argument('-st', '--start-time', nargs='*', help='Start time for the time range to filter the data by. Leave blank or set to \'None\' to not filter by start time. Defaults to \'None\'.')
    parser.add_argument('-et', '--end-time', nargs='*', help='End time for the time range to filter the data by. Leave blank or set to \'None\' to not filter by end time. Defaults to \'None\'.')
    parser.add_argument('-il', '--id-list', nargs='*', help='List of ids to filter the data by. Defaults to [].')
    parser.add_argument('-ifm', '--id-filter-mode', type=str, choices=['include', 'exclude'], help='Mode for filtering by id. Include mode keeps only the ids given for id-list, while exlcude mode keeps all the others. Defaults to \'include\'.')
    args = parser.parse_args()
    print(args)
    if args.start_time != None:
        args.start_time = ' '.join(args.start_time)
    if args.end_time != None:
        args.end_time = ' '.join(args.end_time)

    # Parameter defaults
    defaults = {
        'output': 'subset.csv',
        'start_time': 'None',
        'end_time': 'None',
        'id_list': [],
        'id_filter_mode': 'include'
    }

    # Create parameters dictionary from command line arguments and a config yaml file if provided
    params = vars(args)
    if args.config:
        basepath = os.path.dirname(args.config)
        config = yaml.safe_load(open(args.config, 'r'))
        paths = ['input', 'output']
        for path in paths:
            arg_path = path.replace('-', '_')
            if params[arg_path] is None and path in config:
                params[arg_path] = os.path.normpath(os.path.join(basepath, config[path]))
        opt_args = ['start-time', 'end-time', 'id-list', 'id-filter-mode']
        for name in opt_args:
            arg_name = name.replace('-', '_')
            if params[arg_name] is None and name in config:
                params[arg_name] = config[name]
    for key, val in defaults.items():
        if params[key] is None:
            params[key] = val
    print('params:', params)

    # Read data from input file
    shark_df = pd.read_csv(params['input'])
    
    # Filter data
    if params['start_time'].lower() in ['', 'none']:
        start_time = None
    else:
        start_time = datetime.fromisoformat(params['start_time'])
    if params['end_time'].lower() in ['', 'none']:
        end_time = None
    else:
        end_time = datetime.fromisoformat(params['end_time'])
    shark_df = filter_id(filter_time(shark_df, start_time=start_time, end_time=end_time), id_list=params['id_list'], mode=params['id_filter_mode'])

    # Write data to output file
    shark_df.to_csv(params['output'], index=False)