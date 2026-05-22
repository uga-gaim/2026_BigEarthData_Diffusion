import socket

hostname = socket.gethostname()

# This is the folder which includes "derived" and "input" subfolders
if hostname == 'Geog-GeoAI-Lambda-Vector':
    # On UGA Vector
    DATA_ROOT = '/home/myid/wh63884/data'
elif hostname.startswith('exp'):
    # On Expanse
    DATA_ROOT = '/expanse/lustre/scratch/weiming/temp_project/data'
else:
    import re
    sapelo_pattern = '^[a-z][0-9]+-[0-9]+$'

    if bool(re.match(sapelo_pattern, hostname)):
        # On Expanse
        DATA_ROOT = '/work/whlab/data'
    else:
        raise Exception(f'Unknown hostname {hostname}')
