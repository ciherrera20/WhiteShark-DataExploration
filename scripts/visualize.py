import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from shapely.geometry import Point
from pyproj import CRS
from keplergl import KeplerGl
from datetime import timedelta
import argparse
import json
import yaml
import os

def get_receivers_gdf(filename):
    '''Load receiver positions from a CSV into a GeoDataFrame'''
    receivers_gdf = pd.read_csv(filename)
    receivers_gdf = gpd.GeoDataFrame(receivers_gdf[['Station']], geometry=gpd.points_from_xy(receivers_gdf.Lng, receivers_gdf.Lat))
    receivers_gdf = receivers_gdf.set_crs('EPSG:4326')
    return receivers_gdf

def get_receiver_array_size(receivers_gdf):
    '''Calculate the width and height of the receiver array in meters'''
    point = receivers_gdf['geometry'][0]
    max_lon, max_lat = point.coords[0]
    min_lon, min_lat = point.coords[0]
    for index, row in receivers_gdf.iterrows():
        lon, lat = row['geometry'].coords[0]
        max_lon = max(max_lon, lon)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        min_lat = min(min_lat, lat)
    bounds_gdf = gpd.GeoDataFrame([[Point(min_lon, min_lat)], [Point(max_lon, max_lat)]], geometry='geometry', crs={'init': 'epsg:4326'}, columns=['geometry'])
    aeqd = CRS(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=min_lat, lon_0=min_lon).srs
    bounds_gdf = bounds_gdf.to_crs(crs=aeqd)
    width, height = bounds_gdf['geometry'][1].coords[0]
    return width, height

def get_shark_gdf(filename):
    '''Load shark position data into a GeoDataFrame'''
    shark_gdf = pd.read_csv(filename)
    shark_gdf['t'] = pd.to_datetime(shark_gdf['DATETIME'])
    shark_gdf = gpd.GeoDataFrame(shark_gdf[['TRANSMITTER', 't']], geometry=gpd.points_from_xy(shark_gdf.LON, shark_gdf.LAT))
    shark_gdf = shark_gdf.set_crs('EPSG:4326')
    shark_gdf = shark_gdf.set_index('t').tz_localize(None)
    return shark_gdf

def get_map_config(filename):
    '''Load map config json file into python object'''
    if not os.path.isfile(filename):
        print('The given map config file does not exist. Default KeplerGl config will be used.')
        return None
    keyword_dict = {'None': 'null', 'True': 'true', 'False': 'false'}
    map_config_str = open(filename, 'r').read()
    if '{' not in map_config_str:
        print('The given map config file is empty. Default KeplerGl config will be used.')
        return None
    for keyword, sub in keyword_dict.items():
        map_config_str = map_config_str.replace(keyword, sub)
    return json.loads(map_config_str)

def get_trajectories_gdf(traj_col):
    '''Convert a TrajectoryCollection into a GeoDataFrame suitable for KeplerGl'''
    data = []
    for traj in traj_col.trajectories:
        for i, (timestamp, row) in zip(range(traj.df.shape[0]), traj.df.iterrows()):
            if i == 0:
                continue
            point = row['geometry']
            prev_point = traj.df['geometry'][i - 1]
            x, y = point.coords[0]
            prev_x, prev_y = prev_point.coords[0]
            data.append([str(timestamp), x, y, prev_x, prev_y, row['TRANSMITTER'], point])
    return gpd.GeoDataFrame(data, columns=['t', 'lon', 'lat', 'prev_lon', 'prev_lat', 'TRANSMITTER', 'geometry'])

def traj_contains_time(traj, time):
    '''Returns whether or not a time falls between a trajectory's start and end time'''
    return traj.get_start_time() <= time <= traj.get_end_time()

def get_interpolated_gdf(traj_col, num_points=100):
    '''Create some interpolated trajectories from the given trajectory collection'''
    start_time = min([traj.get_start_time() for traj in traj_col.trajectories])
    end_time = min([traj.get_end_time() for traj in traj_col.trajectories])
    delta = (end_time - start_time) / num_points

    data = []
    for traj in traj_col.trajectories:
        transmitter = traj.id.split('_')[0]
        point_dict = {}
        for i in range(1, num_points):
            prev_frame_time = start_time + (delta * (i - 1))
            frame_time = prev_frame_time + delta
            if traj_contains_time(traj, prev_frame_time) and traj_contains_time(traj, frame_time):
                # Calculate interpolated point
                point = traj.get_position_at(frame_time, method='interpolated')
                point_dict[frame_time] = point
                x, y = point.coords[0]

                # Retrieve previous interpolated point
                if prev_frame_time not in point_dict:
                    prev_point = traj.get_position_at(frame_time, method='interpolated')
                    point_dict[prev_frame_time] = prev_point
                else:
                    prev_point = point_dict[prev_frame_time]
                prev_x, prev_y = prev_point.coords[0]

                # Append data
                data.append([str(frame_time), x, y, prev_x, prev_y, transmitter, point])
    return gpd.GeoDataFrame(data, columns=['t', 'lon', 'lat', 'prev_lon', 'prev_lat', 'TRANSMITTER', 'geometry'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a KeplerGl map visualization of shark telemetry data.')
    parser.add_argument('positions', nargs='?', help='CSV file containing the telemetry data to visualize.')
    parser.add_argument('receivers', nargs='?', help='CSV file containing the receiver locations.')
    parser.add_argument('output', nargs='?', help='Name of the output html map file. Defaults to \'./map.html\'.')
    parser.add_argument('-c', '--config', help='Name of the config yaml file to use. All of the other arguments can be provided there using the long argument name without the starting dashes.')
    parser.add_argument('-mc', '--map-config', help='JSON file containing the map configuration given to KeplerGl.')
    parser.add_argument('-nip', '--num-interpolated-points', type=int, help='Number of interpolated points to calculate. Defaults to 100.')
    parser.add_argument('-mxd', '--max-distance', type=float, help='Maximum distance (in meters) between significant points when aggregating trajectories. Defaults to 10.0.')
    parser.add_argument('-mnd', '--min-distance', type=float, help='Minimum distance (in meters) between significant points when aggregating trajectories. Defaults to 1.0.')
    parser.add_argument('-msd', '--min-stop-duration', type=int, help='Minimum duration (in seconds) required for stop detection when aggregating trajectories. Defaults to 3600.')
    parser.add_argument('-ma', '--min-angle', type=float, help='Minimum angle (in degrees) for significant point extraction when aggregating trajectories. Defaults to 45.0.')
    args = parser.parse_args()

    # Parameter defaults
    defaults = {
        'output': 'map.html',
        'num_interpolated_points': 100,
        'max_distance': 100.0,
        'min_distance': 10.0,
        'min_stop_duration': 3600,
        'min_angle': 45
    }

    # Create parameters dictionary from command line arguments and a config yaml file if provided
    params = vars(args)
    if args.config:
        basepath = os.path.dirname(args.config)
        config = yaml.safe_load(open(args.config, 'r'))
        paths = ['positions', 'receivers', 'output', 'map-config']
        for path in paths:
            arg_path = path.replace('-', '_')
            if params[arg_path] is None and path in config:
                params[arg_path] = os.path.normpath(os.path.join(basepath, config[path]))
        opt_args = ['num-interpolated-points', 'max-distance', 'min-distance', 'min-stop-duration', 'min-angle']
        for name in opt_args:
            arg_name = name.replace('-', '_')
            if params[arg_name] is None and name in config:
                params[arg_name] = config[name]
    for key, val in defaults.items():
        if params[key] is None:
            params[key] = val
    print('params:', params)

    # Read data from files
    receivers_gdf = get_receivers_gdf(params['receivers'])
    width, height = get_receiver_array_size(receivers_gdf)
    print('Receiver array is', width, 'meters wide and', height, 'meters high.')
    shark_gdf = get_shark_gdf(params['positions'])
    if params['map_config']:
        map_config = get_map_config(params['map_config'])
    else:
        map_config = None

    # Process data
    shark_trajs = mpd.TrajectoryCollection(shark_gdf, 'TRANSMITTER')
    shark_trajs_agg = mpd.TrajectoryCollectionAggregator(shark_trajs, params['max_distance'], params['min_distance'], timedelta(seconds=params['min_stop_duration']), min_angle=params['min_angle'])
    trajectories_gdf = get_trajectories_gdf(shark_trajs)
    significant_points_gdf = shark_trajs_agg.get_significant_points_gdf()
    significant_points_gdf['lon'] = [point.coords[0][0] for point in significant_points_gdf['geometry']]
    significant_points_gdf['lat'] = [point.coords[0][1] for point in significant_points_gdf['geometry']]
    interpolated_gdf = get_interpolated_gdf(shark_trajs, num_points=params['num_interpolated_points'])

    # Create KeplerGl instance and add data
    m = KeplerGl(height=600)
    m.add_data(receivers_gdf.copy(), 'receivers')
    m.add_data(trajectories_gdf.copy(), 'trajectories')
    m.add_data(significant_points_gdf.copy(), 'significant points')
    m.add_data(shark_trajs_agg.get_clusters_gdf().copy(), 'clusters')
    m.add_data(shark_trajs_agg.get_flows_gdf().copy(), 'flows')
    m.add_data(interpolated_gdf.copy(), 'interpolated')
    m.save_to_html(file_name=params['output'], config=map_config)

    # Give the option to re-save the map
    print('')
    re_save = True
    while (re_save):
        print('Would you like to save the map again with a new configuration to avoid having to re-process the data?')
        ans = input('(y/n), or enter a path to a new map config file: ')
        if os.path.isfile(ans):
            map_config = get_map_config(ans)
            print('Using config file', ans)
            m.save_to_html(file_name=params['output'], config=map_config)
        elif ans.lower() in ['yes', 'y', 'yea', 'ye', 'yep', 'ok', 'sounds good']:
            map_config = get_map_config(params['map_config'])
            print('Using original config file')
            m.save_to_html(file_name=params['output'], config=map_config)
        elif ans.lower() in ['no', 'n', 'nope', 'nop', 'exit', ':q']:
            re_save = False
        else:
            print('The given path is not a file')
        print('')