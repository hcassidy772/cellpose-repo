from stardist import (
    calculate_extents,
    Rays_GoldenSpiral,
    fill_label_holes,
    random_label_cmap,
)
from stardist.models import StarDist3D, Config3D
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import normalize
from pathlib import Path
import tensorflow as tf
from util import merge
import torch

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


gpu_check = torch.cuda.is_available()

if not gpu_check:
    print("gpu not available")
    print("ending")
    raise Exception

# -===- data prep -===-


raw_dir = "/mnt/parscratch/users/ach22jc/tifs/sdist-raw/"
mas_dir = "/mnt/parscratch/users/ach22jc/tifs/sdist-mas/"

X_paths = sorted((Path(raw_dir).glob("*.tif")))
Y_paths = sorted((Path(mas_dir).glob("*.tif")))

X = [imread(x) for x in X_paths]
Y = [imread(y) for y in Y_paths]

X = [merge(x) for x in X]
X = [normalize(i, 1, 99.8, axis=(0, 1, 2)) for i in X]

if len(X) != len(Y):
    print("diff number of tifs in raw/mask")
    print("raw:")
    print(len(X))
    print("mask:")
    print(len(Y))
    print("ending")
    raise Exception

for i in range(len(X)):
    x = X[i]
    y = Y[i]
    if x.ndim != y.ndim:
        print('merge bad')
        print(x.shape)
        print(y.shape)
        raise Exception
    if x.shape != y.shape:
        print("shape doesnt match")
        raise Exception
    if X_paths[i].name != Y_paths[i].name:
        print("name doesnt match")
        raise Exception

X = [x[2:-2] for x in X]
Y = [y[2:-2] for y in Y]

rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]

Y = [fill_label_holes(y) for y in Y]

X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

X_val_path, Y_val_path = [X_paths[i] for i in ind_val], [Y_paths[i] for i in ind_val]

print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))

# -===- model prep -===-

gpu = True
extents = calculate_extents(Y)
anisotropy = tuple(np.max(extents) / extents)
rays = Rays_GoldenSpiral(96, anisotropy)
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)


conf = Config3D(
    use_gpu=gpu,
    rays=rays,
    grid=grid,
    anisotropy=anisotropy,
    train_patch_size=(12, 480, 480),
    train_batch_size=2,
    train_epochs=200,
)

model = StarDist3D(conf, name="model-5-s", basedir="models")


# -===- training -===-
def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y


history = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)

# -===- optimisation -===-
masks = []
for i in range(len(Y_val_path)):
    masks.append(imread(Y_val_path[i]))

Y_val_pred = [
    model.predict_instances(
        x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False
    )
    for x in X_val
]

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
lbl_cmap = random_label_cmap()
for i in range(3):
    ax[i, 0].imshow(X_val[i][8, :, :], cmap="gray")
    ax[i, 0].set_title(f"Validation Image {i + 1}")
    ax[i, 1].imshow(Y_val_pred[i][0][8, :, :], cmap=lbl_cmap)
    ax[i, 1].set_title(f"StarDist Prediction {i + 1}")
    ax[i, 2].imshow(masks[i][8, :, :], cmap=lbl_cmap)
    ax[i, 2].set_title(f"Input Mask {i + 1}")

fig.savefig("stardist_predictions.png", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history["loss"], label="train")
ax.plot(history.history["val_loss"], label="validation")
ax.plot(history.history["dist_dist_iou_metric"], label="train dist iou")
plt.legend()

fig.savefig("loss.png", bbox_inches="tight")
