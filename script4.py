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

# dev = torch.cuda.device(0)

# 4.0
# path = '/users/ach22jc/.cellpose/models/bact_phase_cp3'
model = models.CellposeModel(gpu=True)
# model = models.CellposeModel(model_type="bact_phase_cp3", gpu=True)

gnome = Path("/users/ach22jc/max.tif")
tif = imread(gnome)

# base values
diameter = 20
min_size = 12
cellprob_threshold = 5
flow_threshold = 0.8
flow3D_smooth = 2

print(tif.shape)

print('running diam')
for i in range(1, 21):  # dont run diam 0 you spanner
    diam = 2 * i
    mask, two, three = model.eval(
        tif,
        do_3D=True,
        z_axis=0,
        diameter=diam,
        # cellprob_threshold=cellprob_threshold,
        # flow_threshold=flow_threshold,
        # min_size=min_size,
        # flow3D_smooth=flow3D_smooth
    )
    outstr = "/users/ach22jc/test-outputs/cp4/diam/" + str(i) + ".tif"
    imwrite(outstr, mask)

print('running cellprob_threshold')
for i in range(11):
    cpt = i / 10
    mask, two, three = model.eval(
        tif,
        do_3D=True,
        z_axis=0,
        # diameter=diameter,
        cellprob_threshold=cpt,
        # flow_threshold=flow_threshold,
        # min_size=min_size,
        # flow3D_smooth=flow3D_smooth
    )
    outstr = "/users/ach22jc/test-outputs/cp4/cpt/" + str(i) + ".tif"
    imwrite(outstr, mask)

print('running flow_threshold')
for i in range(11):
    ft = i / 10
    mask, two, three = model.eval(
        tif,
        do_3D=True,
        z_axis=0,
        # diameter=diameter,
        # cellprob_threshold=cellprob_threshold,
        flow_threshold=ft,
        # min_size=min_size,
        # flow3D_smooth=flow3D_smooth
    )
    outstr = "/users/ach22jc/test-outputs/cp4/ft/" + str(i) + ".tif"
    imwrite(outstr, mask)

print('running min_size')
for i in range(1, 11):
    ms = 5 * i
    mask, two, three = model.eval(
        tif,
        do_3D=True,
        z_axis=0,
        # diameter=diameter,
        # cellprob_threshold=cellprob_threshold,
        # flow_threshold=flow_threshold,
        min_size=ms,
        # flow3D_smooth=flow3D_smooth
    )
    outstr = "/users/ach22jc/test-outputs/cp4/ms/" + str(i) + ".tif"
    imwrite(outstr, mask)


print('running flow3D_smooth')
for i in range(10):
    f3d = 2 * i
    mask, two, three = model.eval(
        tif,
        do_3D=True,
        z_axis=0,
        # diameter=diameter,
        # cellprob_threshold=cellprob_threshold,
        # flow_threshold=flow_threshold,
        # min_size=min_size,
        flow3D_smooth=f3d
    )
    outstr = "/users/ach22jc/test-outputs/cp4/f3d/" + str(i) + ".tif"
    imwrite(outstr, mask)
print('tada')
