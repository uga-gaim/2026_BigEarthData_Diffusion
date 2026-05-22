import numpy as np

from herbie import Herbie
from datetime import timedelta
from scipy.interpolate import griddata


def download_hrrr(valid_time, lead_time, grids_x, grids_y):

    if lead_time == 1:
        
        H = Herbie(
            str(valid_time - timedelta(days=1)) + " 12:00",
            model="hrrr",
            product="sfc",
            fxx=24,
            verbose=False,
        )
        
        H = H.xarray(":APCP:")
    else:
        
        H_0 = Herbie(
            str(valid_time - timedelta(days=lead_time)) + " 12:00",
            model="hrrr",
            product="sfc",
            fxx=24 * lead_time,
            verbose=False,
        ).xarray(":APCP:")
        
        H_1 = Herbie(
            str(valid_time - timedelta(days=lead_time)) + " 12:00",
            model="hrrr",
            product="sfc",
            fxx=24 * (lead_time - 1),
            verbose=False,
        ).xarray(":APCP:")

        H = H_0 - H_1
    
    
    # Get the HRRR lat/lon and precipitation values
    hrrr_lon = H.longitude.values.flatten() - 360
    hrrr_lat = H.latitude.values.flatten()
    hrrr_precip = H.tp.values.flatten()
    
    # Create target grid mesh
    target_lon, target_lat = np.meshgrid(grids_x, grids_y)
    
    # Resample using nearest neighbor
    precip = griddata(
        points=(hrrr_lon, hrrr_lat),
        values=hrrr_precip,
        xi=(target_lon, target_lat),
        method='nearest'
    )
    
    return precip
