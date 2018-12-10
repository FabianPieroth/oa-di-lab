
print('----------- start importing ------------')
print('import numpy')
import numpy as np
print('numpy version ' + np.__version__)

print('import scipy')

import scipy
print('scipy version ' + scipy.__version__)
#import scipy.io
#from scipy.ndimage.interpolation import map_coordinates
#from scipy import interpolate as ipol

print('import skimage')
import skimage
print('skimage version ' + skimage.__version__)
#from skimage.transform import resize
#import skimage.filters as skf

print('import torch')
import torch
print('torch version ' + torch.__version__)
#from torch.nn import Conv2d
#from torch.nn import ConvTranspose2d
#from torch import nn
#from torch.nn.functional import relu



# probably in base python3.6
print('import others')
import os
import sys
from pathlib import Path
import pickle
import random


print('Done importing')

print('------------ Testing CUDA ----------')
if torch.cuda.is_available():
    print('CUDA available')
    cur_dev = torch.cuda.current_device()
    print('current device ' + str(cur_dev))
    print('device count ' + str(torch.cuda.device_count()))
    print('device name ' + torch.cuda.get_device_name(cur_dev))
else:
    print('CUDA not available')

print('------------ Goodbye ---------------')