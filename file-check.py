from util import merge, check_sim
from pathlib import Path
from tifffile import imread

raw = sorted(list(Path('/mnt/parsratch/users/ach22jc/tifs/sdist-raw').glob('*.tif')))
mas = sorted(list(Path('/mnt/parsratch/users/ach22jc/tifs/sdist-mas').glob('*.tif')))

raw = [merge(imread(i)) for i in raw]
mas = [imread(i) for i in mas]

for i in range(len(raw)):
    if raw[i].shape != mas[i].shape:
        raise Exception
