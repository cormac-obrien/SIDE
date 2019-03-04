import h5py
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import requests
from scipy.io import loadmat
from tqdm import tqdm

DATA_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
DATA_FILE = 'nyu_depth_v2_labeled.mat'
SPLIT_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
SPLIT_FILE = 'splits.mat'

NYU_WIDTH = 640
NYU_HEIGHT = 480
NYU_CHANNELS = 3

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

    return h5py.File(DATA_FILE, 'r'), loadmat(SPLIT_FILE)

def get_image(images, index):
    return np.transpose(images[index] / 255.0, (2, 1, 0))

def get_depth(depths, index):
    return np.transpose(depths[index], (1, 0))

class NyuSequence(Sequence, batch_size=1, shuffle=True, dims=(NYU_WIDTH, NYU_HEIGHT)):
    def __init__(self):
        data, split = get_data()
        self.train_ids = split['trainNdxs']
        self.images = data['images']
        self.depths = data['depths']
        self.batch_size = batch_size
        self.state = np.random.RandomState(seed=42)
        self.dims

        self.shuffle = shuffle
        if shuffle:
            self.perm = self.state.permutation(len(self.train_ids))
        else:
            self.perm = np.arange(len(self.train_ids))

    def __len__(self):
        return int(np.ceil(len(self.train_ids) / batch_size))

    def __getitem__(self, index):
        perm_ids = self.perm[index * self.batch_size : (index + 1) * self.batch_size]
        batch_ids = [self.train_ids[i] for i in perm_ids]
        X, y = self._generate(batch_ids)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.perm = self.state.permutation(len(self.train_ids))

    def _generate(self, ids):
        w, h = self.dims
        X = np.empty((self.batch_size, h, w, NYU_CHANNELS))
        y = np.empty((self.batch_size, h, w, 1))

        for i in range(self.batch_size):
            X[i,] = get_image(self.images, i)
            y[i] = get_depth(self.depths, i)

        return X, y