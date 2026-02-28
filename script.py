from cellpose import models
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import torch

check = torch.cuda.is_available()

if not check:
    print('gpu not available')
    print('ending')
    raise Exception

# dev = torch.cuda.device(0)

# 4.0
path = '/users/ach22jc/.cellpose/models/bact_phase_cp3'
model = models.CellposeModel(pretrained_model=path, gpu=True)
# model = models.CellposeModel(model_type="bact_phase_cp3", gpu=True)

gnome = Path("/users/ach22jc/test.tif")
tif = imread(gnome)
tif = np.moveaxis(tif, 1, -1)
tif = np.max(tif, axis=3)

# paramaters to play with later
# anisotropy = 488 / 19 # this works bad
# diameter = 20  # used to resample the image? wild try it out

print(tif.shape)

flow = 5
for i in range(20):
    # flow3D_smooth isnt implemented on 3.0, check more versions
    mask, two, three = model.eval(
        tif, do_3D=True, z_axis=0, cellprob_threshold=-6, flow_threshold=10, min_size=i
    )
    print(str(i) + " done")

    outstr = "/users/ach22jc/test-outputs/out-flow" + str(i) + ".tif"
    imwrite(outstr, mask)

