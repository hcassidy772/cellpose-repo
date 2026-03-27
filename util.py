import numpy as np


def merge(tif):
    if tif.ndim == 4:
        tif = np.max(tif, axis=1)
    return tif


def check_sim(t1, t2):
    return np.array_equal(t1, t2)


def vol(diam):
    return int((4 / 3) * (3.14) * ((diam / 2) ** 3))  # eq for sphere vol
