import io
import pickle
import zipfile
import requests
import tempfile

import numpy as np
import geopandas as gpd

from pathlib import Path
from tqdm.auto import tqdm
from shapely.geometry import Point
from shapely.ops import unary_union, polygonize


def land(x, y, save_file):
    """
    Return: An mask array with dimesion [y, x]
    """
    
    if Path(save_file).exists():
        with open(save_file, 'rb') as obj:
            on_land_mask = pickle.load(obj)
    else:
        url = (
            'https://naciscdn.org/naturalearth'
            '/110m/physical/ne_110m_coastline.zip'
        )
        
        tmp_folder = Path(tempfile.gettempdir())
        
        # Access coastline data
        response = requests.get(url)
        zip_bytes = io.BytesIO(response.content)
        
        with zipfile.ZipFile(zip_bytes) as z:
            z.extractall(tmp_folder)
        
        gdf_lines = gpd.read_file(tmp_folder / 'ne_110m_coastline.shp')
        
        # Convert line to polygon
        merged_lines = unary_union(gdf_lines.geometry)
        polygons = list(polygonize(merged_lines))
        gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=gdf_lines.crs)
        
        x2d, y2d = np.meshgrid(x, y)
        points = [Point(xy) for xy in zip(x2d.ravel(), y2d.ravel())]
        on_land = np.array([
            gdf_polygons.contains(pt).any()
            for pt in tqdm(points)
        ])
        
        on_land_mask = on_land.reshape(y2d.shape)

        with open(save_file, 'wb') as obj:
            pickle.dump(on_land_mask, obj)

    return on_land_mask
    