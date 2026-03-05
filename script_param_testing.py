from cellpose import models
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import torch
import logging

check = torch.cuda.is_available()

logging.basicConfig(filename="ms_run.log", level=logging.INFO)

if not check:
    print("gpu not available")
    print("ending")
    raise Exception

# ========== cellpose setup ==========

# 4.0
model = models.CellposeModel(gpu=True)

# gnome = Path("/users/ach22jc/test-images/")
gnome = Path("/users/ach22jc/hnt.tif")
tif = imread(gnome)
# tifs = gnome.glob("*.tif")

# base values
# diameter = 20
# min_size = 12 # NUMBER OF VOXELS not diameter
# cellprob_threshold = 5
# flow_threshold = 0.1  # doesnt work for 3D
flow3D_smooth = 2
min_diam = 18

# ========== other setup ==========


def vol(diam):
    return (4 / 3) * (3.14) * ((diam / 2) ** 3)  # eq for sphere vol


# ========== for loop ==========

for i in range(10,30):
    if tif.ndim == 4:
        tif = np.max(tif, axis=1)
    mask, two, three = model.eval(
        tif, do_3D=True, z_axis=0, flow3D_smooth=flow3D_smooth,
        min_size=vol(i)
    )
    outstr = "/users/ach22jc/test-outputs/cp4/hnt/ft/" + (str(i)) + '.tif'
    imwrite(outstr, mask)


# for i in range():
#     if tif.ndim == 4:
#         tif = np.max(tif, axis=1)
#     mask, two, three = model.eval(
#         tif, do_3D=True, z_axis=0, flow3D_smooth=flow3D_smooth
#     )
#     outstr = "/users/ach22jc/test-outputs/cp4/bulk/" + (i.name)
#     imwrite(outstr, mask)

print("tada")
