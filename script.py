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

model = models.CellposeModel(model_type="bact_phase_cp3", gpu=True)
gnome = Path("/users/ach22jc/test.tif")

# paramaters to play with later
# anisotropy = 488 / 19 # this works bad
# diameter = 20  # used to resample the image? wild try it out

tif = imread(gnome)

# merge channels (max not mean)
print(tif.shape)
tif = np.moveaxis(tif, 1, -1)
tif = np.max(tif, axis=3)
print(tif.shape)

flow = 5
for i in range(-6, 7):
    mask, two, three = model.eval(
        tif, do_3D=True, z_axis=0, flow3D_smooth=flow, cellprob_threshold=i
    )
    print(str(i) + " done")

    outstr = "/users/ach22jc/test-outputs/out-standard" + str(i) + ".tif"
    imwrite(outstr, mask)

