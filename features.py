import numpy as np
import scipy as sp
import skimage
import skimage.feature
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

haralick_props = [
    'contrast',
    'dissimilarity',
    'homogeneity',
    'energy',
    'correlation',
    'ASM'
]

# split a 2-D array into patches
def patches_2d(y, shape):
    h, w = np.shape(y)
    ph, pw = shape

    # (img_r, patch_r, img_c, patch_c)
    reshaped = y.reshape(h // ph, ph, w // pw, pw)

    # (img_r, img_c, patch_r, patch_c)
    return np.transpose(reshaped, (0, 2, 1, 3))

def patch_glcm(rgb_hwc, shape, dist, angle):
    h, w, _ = np.shape(rgb_hwc)
    ycbcr = skimage.color.rgb2ycbcr(rgb_hwc)
    y = ycbcr[:, :, 0]
    bins = np.linspace(0, 256, num=8)
    quant = np.digitize(y, bins)
    patches = patches_2d(quant, shape)

    # image row, image col, patch glcm row, patch glcm col, angle
    out = np.zeros((patches.shape[0], patches.shape[1], 8, 8, 1, 4))

    for py in range(patches.shape[0]):
        for px in range(patches.shape[1]):
            glcm = skimage.feature.greycomatrix(patches[py, px], [dist], angle, levels=len(bins))
            out[py, px, ] = glcm

    # sum over angles
    return np.sum(out, axis=5)

def patch_haralick(glcms):
    # imrow, imcol, prop
    out = np.zeros((glcms.shape[0], glcms.shape[1], len(haralick_props)))
    for py in range(glcms.shape[0]):
        for px in range(glcms.shape[1]):
            # imrow, imcol, glcmrow, glcmcol, dist, (dummy angle)
            glcm = np.expand_dims(glcms[py, px, :, :], -1)
            for i, prop in enumerate(haralick_props):
                out[py, px, i] = skimage.feature.greycoprops(glcm, prop)[0, 0]

    return out

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
