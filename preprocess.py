import features
import h5py
import numpy as np
import nyu
import os
import skimage.transform
from tqdm import tqdm

# Channel IDs
CHAN_R = 0
CHAN_G = 1
CHAN_B = 2
CHAN_Y = 3
CHAN_CB = 4
CHAN_CR = 5
CHAN_LAWS_LL = 6
CHAN_LAWS_LE = 7
CHAN_LAWS_LS = 8
CHAN_LAWS_EL = 9
CHAN_LAWS_EE = 10
CHAN_LAWS_ES = 11
CHAN_LAWS_SL = 12
CHAN_LAWS_SE = 13
CHAN_LAWS_SS = 14
CHAN_NB_0 = 15
CHAN_NB_30 = 16
CHAN_NB_60 = 17
CHAN_NB_90 = 18
CHAN_NB_120 = 19
CHAN_NB_150 = 20
CHAN_HARALICK_4X_CONTRAST = 21
CHAN_HARALICK_4X_DISSIMILARITY = 22
CHAN_HARALICK_4X_HOMOGENEITY = 23
CHAN_HARALICK_4X_ENERGY = 24
CHAN_HARALICK_4X_CORRELATION = 25
CHAN_HARALICK_4X_ASM = 26
CHAN_HARALICK_8X_CONTRAST = 27
CHAN_HARALICK_8X_DISSIMILARITY = 28
CHAN_HARALICK_8X_HOMOGENEITY = 29
CHAN_HARALICK_8X_ENERGY = 30
CHAN_HARALICK_8X_CORRELATION = 31
CHAN_HARALICK_8X_ASM = 32
CHAN_HARALICK_16X_CONTRAST = 33
CHAN_HARALICK_16X_DISSIMILARITY = 34
CHAN_HARALICK_16X_HOMOGENEITY = 35
CHAN_HARALICK_16X_ENERGY = 36
CHAN_HARALICK_16X_CORRELATION = 37
CHAN_HARALICK_16X_ASM = 38
N_CHANNELS = CHAN_HARALICK_16X_ASM + 1

NEAREST = 0
BILINEAR = 1
BIQUADRATIC = 2
BICUBIC = 3

IMG_W = 640
IMG_H = 480

# largest haralick patch size is 16x16
TRIM_X = 16
TRIM_Y = 16

# trim edges of image to remove dead pixels, then resize to original dims
def trim(a, trim_x=TRIM_X, trim_y=TRIM_Y):
    shape = (IMG_H, IMG_W) + a.shape[2:]
    print(shape)
    trimmed = a[trim_x:-trim_x, trim_y:-trim_y,]
    return skimage.transform.resize(trimmed, shape, mode='reflect', order=BICUBIC)

def feature_maps(img):
    out = np.zeros((480, 640, N_CHANNELS))
    rgb = trim(img)
    print(rgb.dtype)
    ycbcr = skimage.color.rgb2ycbcr(rgb)
    print(ycbcr.dtype)
    laws = trim(features.laws(img))
    print(laws.dtype)
    nb = trim(features.edge(img))
    print(nb.dtype)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    haralick4 = trim(features.patch_haralick(img, (4, 4), 1, angles), trim_x=4, trim_y=4)
    print(haralick4.dtype)
    haralick8 = trim(features.patch_haralick(img, (8, 8), 2, angles), trim_x=2, trim_y=2)
    print(haralick8.dtype)
    haralick16 = trim(features.patch_haralick(img, (16, 16), 4, angles), trim_x=1, trim_y=1)
    print(haralick16.dtype)

    out[:, :, CHAN_R:CHAN_B+1] = rgb
    out[:, :, CHAN_Y:CHAN_CR+1] = ycbcr
    out[:, :, CHAN_LAWS_LL:CHAN_LAWS_SS+1] = laws
    out[:, :, CHAN_NB_0:CHAN_NB_150+1] = nb
    out[:, :, CHAN_HARALICK_4X_CONTRAST:CHAN_HARALICK_4X_ASM+1] = haralick4
    out[:, :, CHAN_HARALICK_8X_CONTRAST:CHAN_HARALICK_8X_ASM+1] = haralick8
    out[:, :, CHAN_HARALICK_16X_CONTRAST:CHAN_HARALICK_16X_ASM+1] = haralick16

# generate per-patch grey-level co-occurrence matrices for all images in the dataset
# - 4x4 patches: D = 1
# - 8x8 patches: D = 2
# - 16x16 patches: D = 4
# angles are 0, pi/4, pi/2, 3pi/4
def gen_glcms():
    path = 'nyuv2_glcms.h5'
    if os.path.exists(path):
        return

    data, splits = nyu.get_data()
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    with h5py.File(path, 'a') as f:
        count = len(data['images'])
        for scale in [4, 8, 16]:
            glcms = f.create_dataset('glcms_{}x'.format(scale), (count, IMG_H // scale, IMG_W // scale, 8, 8, 1))
            for i in tqdm(range(count)):
                img = np.transpose(data['images'][i], (2, 1, 0))
                glcms[i,] = features.patch_glcm(img, (scale, scale), scale / 4, angles)

def gen_haralick(scale):
    path = 'nyuv2_haralick.h5'
    data, splits = nyu.get_data()

    with h5py.File('nyuv2_glcms.h5', 'r') as glcms_data:
        glcms = glcms_data['glcms_{}x'.format(scale)]

        with h5py.File(path, 'a') as f:
            name = 'haralick_{}x'.format(scale)
            if name in f.keys():
                return

            count = len(data['images'])
            haralick = f.create_dataset('haralick_{}x'.format(scale), (count, IMG_H // scale, IMG_W // scale, 6))
            for i in tqdm(range(count)):
                img = np.transpose(data['images'][i], (2, 1, 0))
                haralick[i,] = features.patch_haralick(glcms[i])
