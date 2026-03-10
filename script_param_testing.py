from cellpose import models
# from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import torch
import logging

check = torch.cuda.is_available()

logging.basicConfig(filename="param.log", level=logging.INFO)

if not check:
    print("gpu not available")
    print("ending")
    raise Exception

# ========== cellpose setup ==========

# 4.0
model = models.CellposeModel(gpu=True)

tifs = []
# gnome = Path("/users/ach22jc/test-images/")
# gnome = Path("/users/ach22jc/rf470.tif")
# tif = imread(gnome)
tifs.append(imread('/users/ach22jc/atto-l.tif')) # 25
tifs.append(imread('/users/ach22jc/atto-s.tif')) # 15
tifs.append(imread('/users/ach22jc/shh.tif')) # 25-30
tifs.append(imread('/users/ach22jc/shh.tif')) # 25-30
# shh = imread('/users/ach22jc/shl.tif')
# tifs = gnome.glob("*.tif")

# base values
# diameter = 20
# min_size = 12 # NUMBER OF VOXELS not diameter
# cellprob_threshold = 5
# flow_threshold = 0.1  # doesnt work for 3D
flow3D_smooth = 2  # :thumb:
min_diam = 10  # not very useful, good for removing bg noise on larger images but doesnt affect mask quality

# ========== other setup ==========


def vol(diam):
    return int((4 / 3) * (3.14) * ((diam / 2) ** 3))  # eq for sphere vol


def eval(tif, model=model, f3d=0, min_diam=0):
    mask, two, three = model.eval(
        tif, do_3D=True, z_axis=0,
        flow3D_smooth=f3d,
        min_size=vol(min_diam)
    )
    return mask


def merge(tif):
    if tif.ndim == 4:
        tif = np.max(tif, axis=1)
    return tif

# ========== for loop ==========

# for i in range(10, 30):
#     # if tif.ndim == 4:
#     #     tif = np.max(tif, axis=1)
#     mask, two, three = model.eval(
#         tif, do_3D=True, z_axis=0, flow3D_smooth=flow3D_smooth,
#         diam=i
#     )
#     outstr = "/users/ach22jc/test-outputs/cp4/rf470/diam/" + (str(i)) + '.tif'
#     imwrite(outstr, mask)


for i in range(len(tifs)):
    tif = tifs[i]
    merge(tif)
    mask = eval(tif, f3d=4, min_diam=min_diam)

    outstr = "/users/ach22jc/test-outputs/cp4/param/" + str(i) + ".tif"
    imwrite(outstr, mask)

print("tada")
