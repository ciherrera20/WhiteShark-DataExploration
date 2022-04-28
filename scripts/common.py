import pandas as pd
import geopandas as gpd
import numpy as np
import os

def get_shark_gdf(filename):
    '''Load shark position data into a GeoDataFrame'''
    shark_gdf = pd.read_csv(filename)
    shark_gdf['t'] = shark_gdf['DATETIME'] = pd.to_datetime(shark_gdf['DATETIME'])
    shark_gdf = gpd.GeoDataFrame(shark_gdf, geometry=gpd.points_from_xy(shark_gdf.LON, shark_gdf.LAT))
    shark_gdf = shark_gdf.set_crs('EPSG:4326')
    shark_gdf = shark_gdf.set_index('t').tz_localize(None)
    return shark_gdf

def wrap_to_pi(theta):
    return ((theta - np.pi) % (2 * np.pi)) - np.pi

def mkdir(path):
    if not os.path.isdir(path):
        parent, _ = os.path.split(path)
        mkdir(parent)
        os.mkdir(path)

def softplus(x, sharpness=1):
    return np.maximum(x, 0) + np.log(np.exp(-1 * np.abs(sharpness * x)) + 1) / sharpness

def iir_filter(x, ff=1):
    y = [x[0]]
    for i, x_i in enumerate(x[1:]):
        y.append((1 - ff) * y[i] + ff * x_i)
    return np.array(y)