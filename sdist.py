from stardist import StarDist3D, Config3D
from tifffile import irmead
import numpy as np
import matplotlib as plt
from csbdeep.utils import normalize

# --- data prep ---
raw_dirs = ''
mas_dirs = ''

# --- model prep ---

conf = Config3D(
            use_gpu=True,
            train_patch_size=(19, 488, 488)
        )

model = StarDist3D(conf, name='stardist_class', basedir='models')
