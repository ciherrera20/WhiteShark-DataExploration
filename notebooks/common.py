import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import os
import re

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

def find_files(*sources, name=None, extension=None):
    '''
    Recursively retrieve files from the given source directories whose names and/or extensions (full)match the given patterns.
    name: string or regex pattern
    extension: string or regex pattern
    Returns a DirEntry generator
    '''
    # Compile regexes if needed
    if name is None:
        name = re.compile(r'.*')
    elif type(name) is not re.Pattern:
        name = re.compile(name)
    if extension is None:
        extension = re.compile(r'.*')
    elif type(extension) is not re.Pattern:
        extension = re.compile(extension)

    # Keep track of the sources already scanned and the files already found
    memo_table = {}

    def find_files_helper(*sources):
        # Search through each source directoty
        for source in sources:
            # Get all of the contents of the source directory and search them
            entries = os.scandir(source)
            for entry in entries:
                # Check if the entry has already been scanned or matched
                normed = os.path.normpath(entry.path)
                if normed not in memo_table:
                    memo_table[normed] = True
                    # If the current entry is itself a directory, search it recursively
                    if entry.is_dir():
                        yield from find_files_helper(entry)

                    # Otherwise yield entries whose name matches the name pattern and whose extension matches the extension pattern
                    else:
                        # Return only entries that have not already been found
                        filename, fileext = os.path.splitext(entry.name)
                        if name.fullmatch(filename) is not None and \
                        extension.fullmatch(fileext) is not None:
                            yield entry
            entries.close()
    return find_files_helper(*sources)