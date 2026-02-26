from cellpose import models
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
# from csbdeep.utils import normalize

model = models.CellposeModel(model_type="bact_phase_cp3", gpu=True)
gnome = Path("/users/ach22jc/test.tif")

# paramaters to play with later
anisotropy = 7
diameter = 0
diam_mean = 0

tif = imread(gnome)

# merge channels (max not mean)
print(tif.shape)
tif = np.moveaxis(tif, 1, -1)
tif = np.max(tif, axis=3)
print(tif.shape)


for i in range(10):
    flow = 4 * i
    mask, two, three = model.eval(tif, do_3D=True, z_axis=0, flow3D_smooth=flow)
    print(str(i) + " done")

    print(mask.shape)
    outstr = "/users/ach22jc/test-outputs/out" + str(i) + ".tif"
    imwrite(outstr, mask)
