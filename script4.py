from cellpose import models
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import torch

check = torch.cuda.is_available()

if not check:
    print("gpu not available")
    print("ending")
    raise Exception

# dev = torch.cuda.device(0)

# 4.0
# path = '/users/ach22jc/.cellpose/models/bact_phase_cp3'
model = models.CellposeModel(gpu=True)
# model = models.CellposeModel(model_type="bact_phase_cp3", gpu=True)

gnome = Path("/users/ach22jc/max.tif")
tif = imread(gnome)
# tif = np.moveaxis(tif, 1, -1)
# tif = np.max(tif, axis=3)

# paramaters to play with later
# anisotropy = 488 / 19 # this works bad
diameter = 20
min_size = 12
cellprob_threshold = 5
flow_threshold = 0.8
flow3D_smooth = 2

print(tif.shape)

# for i in range(20):
    # flow3D_smooth isnt implemented on 3.0, check more versions
mask, two, three = model.eval(
    tif,
    do_3D=True,
    z_axis=0,
    diameter=diameter,
    cellprob_threshold=cellprob_threshold,
    flow_threshold=flow_threshold,
    min_size=min_size,
    flow3D_smooth=flow3D_smooth
)
    # print(str(i) + " done")

outstr = "/users/ach22jc/test-outputs/cp4/out.tif"
imwrite(outstr, mask)
