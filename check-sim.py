import numpy as np
from tifffile import imread
from pathlib import Path

t1 = imread(Path('/home/glow/cellpose-outputs/cp4/ft/0.tif'))
t2 = imread(Path('/home/glow/cellpose-outputs/cp4/ft/5.tif'))

print(t1.shape)
print(t2.shape)

if np.array_equal(t1, t2):
    print('identical')
