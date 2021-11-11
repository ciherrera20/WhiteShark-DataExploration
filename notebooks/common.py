import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta

def get_shark_gdf(filename):
    '''Load shark position data into a GeoDataFrame'''
    shark_gdf = pd.read_csv(filename)
    shark_gdf['t'] = shark_gdf['DATETIME'] = pd.to_datetime(shark_gdf['DATETIME'])
    # shark_gdf = gpd.GeoDataFrame(shark_gdf[['TRANSMITTER', 't']], geometry=gpd.points_from_xy(shark_gdf.LON, shark_gdf.LAT))
    shark_gdf = gpd.GeoDataFrame(shark_gdf, geometry=gpd.points_from_xy(shark_gdf.LON, shark_gdf.LAT))
    shark_gdf = shark_gdf.set_crs('EPSG:4326')
    shark_gdf = shark_gdf.set_index('t').tz_localize(None)
    return shark_gdf

def wrap_to_pi(theta):
    return ((theta - np.pi) % (2 * np.pi)) - np.pi