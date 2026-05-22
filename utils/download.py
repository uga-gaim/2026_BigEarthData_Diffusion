import os
import io
import gzip
import zipfile
import requests

from .roots import DATA_ROOT

# The extent extracted for western US from PRISM
west_us_extent = {
    'xmin': -125,
    'xmax': -116,
    'ymin': 32,
    'ymax': 41.5,
}

ga_extent = {
    'xmin': -87,
    'xmax': -80,
    'ymin': 29.5,
    'ymax': 36,
}

pjm_extent = {
    'xmin': -92.38745782027421,
    'xmax': -73.21824893242619,
    'ymin': 35.16451338345693,
    'ymax': 42.22907556933419,
}

def touch_prism(dt, variable):
    assert variable in ['ppt', 'tmean'], 'Currently can download ppt or tmean'
    
    root_folder = os.path.join(DATA_ROOT, 'input/PRISM/daily_stable')
    
    url_template = \
        'https://data.prism.oregonstate.edu/daily/' + variable + \
        '/{}/PRISM_' + variable + '_stable_4kmD2_{}_bil.zip'
    
    url = url_template.format(
        dt.strftime(format='%Y'),
        dt.strftime(format='%Y%m%d')
    )
    
    base_name = url.split('/')[-1].rstrip('.zip')
    output_folder = os.path.join(root_folder, base_name)
    
    if not os.path.exists(output_folder):
        response = requests.get(url)

        if response.status_code == 200:
            os.makedirs(output_folder, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                zip_ref.extractall(path=output_folder)
        else:
            print(f'Errored ({url}), reasons: {response.reason}')

    return os.path.join(output_folder, base_name + '.bil')

    
def touch_mrms(dt):
    root_folder = os.path.join(
        DATA_ROOT,
        'input/MRMS/MRMS_MultiSensor_QPE_01H_Pass1',
    )
    
    url_template = (
        'https://noaa-mrms-pds.s3.amazonaws.com/CONUS'
        '/MultiSensor_QPE_01H_Pass1_00.00/{}'
        '/MRMS_MultiSensor_QPE_01H_Pass1_00.00_{}.grib2.gz'
    )
    
    grib_file = os.path.join(
        root_folder,
        dt.strftime(format='%Y%m%d-%H%M%S') + '.grib2',
    )
    
    if not os.path.exists(grib_file):
        url = url_template.format(
            dt.strftime(format='%Y%m%d'),
            dt.strftime(format='%Y%m%d-%H%M%S')
        )
        
        response = requests.get(url)

        if response.status_code == 200:
            with open(grib_file, "wb") as f:
                f.write(gzip.decompress(response.content))
        else:
            print(f'Errored ({url}), reasons: {response.reason}')

    return grib_file