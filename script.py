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

model = models.CellposeModel(model_type="bact_phase_cp3", gpu=True)
gnome = Path("/users/ach22jc/test.tif")

# paramaters to play with later
anisotropy = 488 / 19
diameter = 20  # used to resample the image? wild try it out

tif = imread(gnome)

# merge channels (max not mean)
print(tif.shape)
tif = np.moveaxis(tif, 1, -1)
tif = np.max(tif, axis=3)
print(tif.shape)


# for i in range(10):
    # flow = 4 * i

mask, two, three = model.eval(
    tif, do_3D=True, z_axis=0, flow3D_smooth=10
)
outstr = "/users/ach22jc/test-outputs/out-none.tif"
imwrite(outstr, mask)

mask, two, three = model.eval(
    tif, do_3D=True, z_axis=0, diameter=diameter, flow3D_smooth=10
)
outstr = "/users/ach22jc/test-outputs/out-dia.tif"
imwrite(outstr, mask)

mask, two, three = model.eval(
    tif, do_3D=True, z_axis=0, anisotropy=anisotropy, flow3D_smooth=10
)
outstr = "/users/ach22jc/test-outputs/out-ani.tif"
imwrite(outstr, mask)

mask, two, three = model.eval(
    tif, do_3D=True, z_axis=0, diameter=diameter, anisotropy=anisotropy, flow3D_smooth=10
)
outstr = "/users/ach22jc/test-outputs/out-all.tif"
imwrite(outstr, mask)

    # print(str(i) + " done")
    #
    # print(mask.shape)
    # outstr = "/users/ach22jc/test-outputs/out" + str(i) + ".tif"
    # imwrite(outstr, mask)
