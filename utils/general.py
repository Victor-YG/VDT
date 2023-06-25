import re
import sys
import chardet
import datetime

import numpy as np


def get_time_stamp():
    time = datetime.datetime.now()
    return time.strftime(r"%Y%m%d_%H%M%S")


def read_pfm(filepath):
    '''read input pfm file'''

    with open(filepath, 'rb') as f:
        color, width, height, scale, endian = None, None, None, None, None

        header = f.readline().rstrip()
        encode_type = chardet.detect(header)
        header = header.decode(encode_type['encoding'])
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode(encode_type['encoding']))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip().decode(encode_type['encoding']))
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data, scale