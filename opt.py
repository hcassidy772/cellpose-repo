from pathlib import Path
from tifffile import imread
from stardist.models import StarDist3D
from csbdeep.utils import normalize

from util import merge

model = StarDist3D(None, name='stardist_class', basedir='/user/ach22jc/models/')

raw_dir = "/mnt/parscratch/users/ach22jc/tifs/sdist-raw/"
mas_dir = "/mnt/parscratch/users/ach22jc/tifs/sdist-mas/"

raw = sorted(list(Path(raw_dir).glob('*.tif')))
true = sorted(list(Path(mas_dir).glob('*.tif')))
# pred = sorted(list(Path('m-out/').glob('*.tif')))
# pred = []

raw = [normalize(merge(imread(i))) for i in raw]

optimized_thresholds = model.optimize_thresholds(
        raw, [imread(i) for i in true],
        nms_threshs=[round(0.05 * x, 3) for x in range(1, 20)],
        iou_threshs=[round(0.05 * x, 3) for x in range(1, 20)]
)

print(optimized_thresholds)
print('tada')
