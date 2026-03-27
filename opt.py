from pathlib import Path
from tifffile import imread
from stardist.models import StarDist3D
from csbdeep.utils import normalize

from util import merge

model = StarDist3D(None, name="model-2", basedir="/users/ach22jc/models/")

raw_dir = "/mnt/parscratch/users/ach22jc/tifs/sdist-raw/"
mas_dir = "/mnt/parscratch/users/ach22jc/tifs/sdist-mas/"

raw = sorted(list(Path(raw_dir).glob("*.tif")))
true = sorted(list(Path(mas_dir).glob("*.tif")))
# pred = sorted(list(Path('m-out/').glob('*.tif')))
# pred = []

ind = [0, 1, 30, 31, 56, 57, 76, 77]
raw = [merge(imread(raw[i])) for i in ind]
true = [imread(true[i]) for i in ind]

raw = [normalize(i, 1, 99.8, axis=(0, 1, 2)) for i in raw]


optimized_thresholds = model.optimize_thresholds(
    raw,
    true,
    nms_threshs=[round(0.04 * x, 2) for x in range(10)],
    iou_threshs=[round(0.04 * x, 2) for x in range(5, 16)],
)

print(optimized_thresholds)
print("tada")
