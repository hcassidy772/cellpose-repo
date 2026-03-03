from cellpose import models
from pathlib import Path
# import numpy as np
from tifffile import imread, imwrite
import torch

check = torch.cuda.is_available()

if not check:
    print("gpu not available")
    print("ending")
    raise Exception

# 4.0
model = models.CellposeModel(gpu=True)

gnome = Path("/users/ach22jc/v2/v2/")
flow3D_smooth = 2
for i in gnome.glob('*.tif'):
    tif = imread(i)

    mask, two, three = model.eval(
        tif,
        do_3D=True,
        z_axis=0,
        flow3D_smooth=flow3D_smooth
        )
    outstr = "/users/ach22jc/test-outputs/cp4/v2/" + (i.name[27:29]) + '-cellposed' + ".tif"
    imwrite(outstr, mask)


print('tada')
