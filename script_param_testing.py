from cellpose import models
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import torch
import logging

check = torch.cuda.is_available()

logging.basicConfig(filename="4.log", level=logging.INFO)

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
# min_size = 12
# cellprob_threshold = 5
flow_threshold = 0.1
flow3D_smooth = 2

# ========== for loop ==========

for i in range(11):
    if tif.ndim == 4:
        tif = np.max(tif, axis=1)
    mask, two, three = model.eval(
        tif, do_3D=True, z_axis=0, flow3D_smooth=flow3D_smooth,
        flow_threshold=(i / 10)
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
