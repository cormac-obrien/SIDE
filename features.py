import numpy as np
import scipy as sp
import skimage
from skimage.transform import resize

# 3x3 Laws filters from [Saxena et al. 2007]
L3 = np.array([1, 2, 1]) / 128.0 / .2
E3 = [-1, 0, 1]
S3 = [-1, 2, -1]

laws1d = [L3, E3, S3]
LAWS_2D = []

for f1 in laws1d:
    for f2 in laws1d:
        LAWS_2D.append(np.outer(f1, f2))

# Oriented edge filters from [Nevatia 1980]
NB0 = np.tile([-100, -100, 0, 100, 100], (5, 1)) / 2000
NB30 = np.array([
    [-100, 32, 100, 100, 100],
    [-100, -78, 92, 100, 100],
    [-100, -100, 0, 100, 100],
    [-100, -100, -92, 78, 100],
    [-100, -100, -100, -32, 100],
]) / 2000

NB60 = -NB30.T
NB90 = -NB0.T
NB120 = -np.flipud(NB60)
NB150 = NB120.T

NB = [NB0, NB30, NB60, NB90, NB120, NB150]
FILTERS = LAWS_2D + NB

def apply_filters(rgb_hwc, filters):
    h, w, _ = np.shape(rgb_hwc)
    ycbcr = skimage.color.rgb2ycbcr(rgb_hwc)
    y = ycbcr[:, :, 0]
    maps = np.zeros((h, w, len(filters)))

    for i, filt in enumerate(filters):
        maps[:, :, i] = sp.ndimage.filters.convolve(y, filt)

    return maps

def laws(rgb_hwc):
    return apply_filters(rgb_hwc, LAWS_2D)

def edge(rgb_hwc):
    return apply_filters(rgb_hwc, NB)

def energy(x):
    pass

# we want to output [scale h w c]
def image_feature_maps(rgb_hwc):
    h, w, _ = rgb_hwc.shape
    ycbcr = skimage.color.rgb2ycbcr(rgb_hwc)

    fmaps = []

    for scale in range(3):
        new_h, new_w = h // (3 ** scale), w // (3 ** scale)
        fmaps.append(np.zeros((new_h, new_w, 34)))
        scaled = resize(rgb_hwc, [new_h, new_w], anti_aliasing=True)
        ycbcr = skimage.color.rgb2ycbcr(scaled)
        y = ycbcr[:, :, 0]
        fmaps[scale][:, :, :3] = ycbcr

        for f_i, filt in enumerate(FILTERS):
            fmap = sp.ndimage.filters.convolve(y, filt)
            if np.shape(fmap) != np.shape(y):
                raise ValueError()
            print(np.shape(fmaps[scale]))
            fmaps[scale][:, :, 3 + f_i] = fmap

    return fmaps

def image_feature_vectors(rgb_hwc):
    fmaps = image_feature_maps(rgb_hwc)
