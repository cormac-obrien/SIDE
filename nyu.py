import features
import h5py
import json
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import requests
from scipy.io import loadmat
import skimage
from skimage.transform import resize
from tqdm import tqdm

DATA_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
DATA_FILE = 'nyu_depth_v2_labeled.mat'
SPLIT_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
SPLIT_FILE = 'splits.mat'

NYU_WIDTH = 640
NYU_HEIGHT = 480
NYU_CHANNELS = 3

# calculated by Eigen et al.
IMG_MEAN = 109.31410628
IMG_STDDEV = 76.18328376
DEPTH_MEAN = 2.53434899
DEPTH_STDDEV = 1.22576694
LOGDEPTH_MEAN = 0.82473954
LOGDEPTH_STDDEV = 0.45723134

def download(src, dst):
    r = requests.get(src, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 4096
    received = 0
    total = math.ceil(total_size / block_size)

    with open(dst, 'wb') as f:
        for block in tqdm(r.iter_content(block_size), total=total, unit='KiB', unit_scale=True):
            received += len(block)
            f.write(block)

def get_data():
    if not os.path.isfile(DATA_FILE):
        print('Downloading dataset from {}...'.format(DATA_URL))
        download(DATA_URL, DATA_FILE)
    else:
        print('Found {}'.format(DATA_FILE))

    if not os.path.isfile(SPLIT_FILE):
        print('Downloading train/test split from {}...'.format(SPLIT_URL))
        download(SPLIT_URL, SPLIT_FILE)
    else:
        print('Found {}'.format(SPLIT_FILE))

    splits = loadmat(SPLIT_FILE)

    # indices are one-based, we need zero-based
    splits['trainNdxs'] = splits['trainNdxs'] - 1
    splits['testNdxs'] = splits['testNdxs'] - 1

    return h5py.File(DATA_FILE, 'r'), splits

def get_image(images, index):
    assert not np.any(np.isnan(images[index]))
    assert not np.any(np.isinf(images[index]))
    return np.transpose((images[index] - IMG_MEAN) / IMG_STDDEV, (2, 1, 0))

def get_logdepth(depths, index):
    return np.transpose((np.log(depths[index]) - LOGDEPTH_MEAN) / LOGDEPTH_STDDEV, (1, 0))

def get_depth(depths, index):
    assert not np.any(np.isnan(depths[index]))
    assert not np.any(np.isinf(depths[index]))
    return np.transpose((depths[index] - DEPTH_MEAN) / DEPTH_STDDEV, (1, 0))

# calculate mean and stddev by Welford's algorithm
class Stats():
    def __init__(self):
        self.M = 0
        self.S = 0
        self.old_M = 0
        self.old_S = 0
        self.count = 0

    def push(self, x):
        self.count += 1

        if self.count == 1:
            self.M = x
            self.old_M = x
            self.old_S = 0
        else:
            self.M = self.old_M + (x - self.old_M) / self.count
            self.S = self.old_S + (x - self.old_M) * (x - self.M)

            self.old_M = self.M
            self.old_S = self.S

    def mean(self):
        if self.count > 0:
            return self.M
        else:
            return 0

    def variance(self):
        if self.count > 1:
            return self.S / (self.count - 1)
        else:
            return 0

    def stddev(self):
        return np.sqrt(self.variance())

def calc_stats():
    data, splits = get_data()
    train_ids = splits['trainNdxs'].flatten()

    rs = Stats()
    gs = Stats()
    bs = Stats()
    ds = Stats()

    for train_id in train_ids:
        r, g, b = data['images'][train_id][:,]
        d = data['depths'][train_id]

        r = r.reshape(-1)
        g = g.reshape(-1)
        b = b.reshape(-1)
        d = d.reshape(-1)

        for i in tqdm(range(len(r))):
            rs.push(r[i])
            gs.push(g[i])
            bs.push(b[i])
            ds.push(d[i])

    with open('stats.json', 'w') as f:
        json.dumps({
            'r': {
                'mean': rs.mean(),
                'stddev': rs.stddev(),
            },
            'g': {
                'mean': gs.mean(),
                'stddev': gs.stddev(),
            },
            'b': {
                'mean': bs.mean(),
                'stddev': bs.stddev(),
            },
            'd': {
                'mean': ds.mean(),
                'stddev': ds.stddev(),
            },
        }, f)

    return (rs, gs, bs, ds)

def stddev():
    data, split = get_data()

    # calculate standard deviation for RGB, YCbCr and depth
    r = 0
    r_prev = 0

    for img in data['images']:
        r, g, b = img[:,]

def train_validation_split(val_ratio):
    _, split = get_data()
    ids = split['trainNdxs'].flatten()
    train_count = round((1 - val_ratio) * len(ids))
    train_ids = ids[:train_count]
    val_ids = ids[train_count:]

    return train_ids, val_ids

feature_sizes = {
    'rgb': 3,
    'ycbcr': 3,
    'laws': 9,
    'edge': 6,
}

feature_funcs = {
    'rgb': lambda x: x,
    'ycbcr': skimage.color.rgb2ycbcr,
    'laws': features.laws,
    'edge': features.edge,
}

class NyuSequence(Sequence):
    def __init__(self,
                 ids,
                 batch_size=1,
                 shuffle=True,
                 features=['rgb'],
                 dims=(NYU_WIDTH, NYU_HEIGHT),
                 depth_scale=1,
                 logdepth=True
    ):
        data, split = get_data()
        self.ids = ids
        self.images = data['images']
        self.depths = data['depths']
        self.batch_size = batch_size
        self.state = np.random.RandomState(seed=42)
        self.dims = (dims[1], dims[0])
        self.depth_scale = depth_scale
        self.logdepth = logdepth

        if not set(features).issubset(set(feature_sizes.keys())):
            raise ValueError("Unrecognized feature name")

        self.features = features

        self.shuffle = shuffle
        if shuffle:
            self.perm = self.state.permutation(len(self.ids))
        else:
            self.perm = np.arange(len(self.ids))

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        perm_ids = self.perm[index * self.batch_size : (index + 1) * self.batch_size]
        batch_ids = [self.ids[i] for i in perm_ids]
        X, y = self._generate(batch_ids)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.perm = self.state.permutation(len(self.ids))

    def _generate(self, ids):
        xh, xw = self.dims
        yh, yw = int(xh * self.depth_scale), int(xw * self.depth_scale)

        total_channels = sum([feature_sizes[f] for f in self.features])
        X = np.empty((len(ids), xh, xw, total_channels))
        y = np.empty((len(ids), yh, yw, 1))

        for i in range(len(ids)):
            rgb = resize(get_image(self.images, ids[i]), self.dims)
            chan_ofs = 0
            for feat in self.features:
                n_channels = feature_sizes[feat]
                X[i, :, :, chan_ofs: chan_ofs + n_channels] = feature_funcs[feat](rgb)
                chan_ofs += n_channels

            if self.logdepth:
                y_tmp = get_logdepth(self.depths, ids[i])
            else:
                y_tmp = get_depth(self.depths, ids[i])

            y[i] = np.expand_dims(resize(y_tmp, (yh, yw)), 2)

        return X, y

    def data_shape(self):
        return (self.dims[0], self.dims[1], sum([feature_sizes[f] for f in self.features]))
