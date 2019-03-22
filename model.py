import json
import loss
import numpy as np
import nyu
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Model():
    def __init__(self, name, features):
        self.name = name
        self.features = features
        train_seq, val_seq, model = build_model(features)
        self.train_seq = train_seq
        self.val_seq = val_seq
        self.model = model
        self.history = None
        self.stats = None

    def summary(self):
        self.model.summary()

    def basedir(self):
        return os.path.join('models', self.name)

    def save_model(self):
        if not os.path.exists(self.basedir()):
            os.makedirs(self.basedir())

        with open(os.path.join(self.basedir(), 'model.json'), 'w') as f:
            f.write(self.model.to_json())

    def train(self):
        cb = EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=20,
            restore_best_weights=True)

        self.history = self.model.fit_generator(
            self.train_seq,
            epochs=300,
            verbose=1,
            validation_data=self.val_seq,
            callbacks=[cb])

    def save_weights(self):
        if self.history is None:
            raise RuntimeError("This model has not yet been trained!")

        self.model.save_weights(os.path.join(self.basedir(), 'weights.h5'))

    def save_history(self):
        if self.history is None:
            raise RuntimeError("This model has not yet been trained!")

        with open(os.path.join(self.basedir(), 'history.json'), 'w') as f:
            json.dump(self.history.history, f)

    def test(self):
        if self.history is None:
            raise RuntimeError("This model has not yet been trained!")

        _, splits = nyu.get_data()
        test_ids = splits['testNdxs'].flatten()
        test_generator = nyu.NyuSequence(
            test_ids,
            features=self.features,
            shuffle=False,
            batch_size=1,
            dims=(320, 240),
            depth_scale=.25)

        loss_stats = np.zeros(len(test_generator))
        for i in range(len(test_generator)):
            X, y = test_generator[i]
            loss_stats[i] = self.model.evaluate(x=X, y=y)

        self.stats = {
            'loss': loss_stats.tolist(),
            'mean': np.mean(loss_stats),
            'median': np.median(loss_stats),
            'stddev': np.std(loss_stats)
        }

        print('Mean: {}', self.stats['mean'])
        print('Median: {}', self.stats['median'])
        print('Standard deviation: {}', self.stats['stddev'])

    def save_stats(self):
        if self.stats is None:
            raise RuntimeError("This model has not yet been tested!")

        with open(os.path.join(self.basedir(), 'stats.json'), 'w') as f:
            json.dump(self.stats, f)

def build_model(features):
    _, splits = nyu.get_data()
    train_ids, val_ids = nyu.train_validation_split(.05)
    train_seq = nyu.NyuSequence(
        train_ids,
        features=features,
        batch_size=4,
        shuffle=False,
        dims=(320, 240),
        depth_scale=.25)
    val_seq = nyu.NyuSequence(
        val_ids,
        features=features,
        batch_size=1,
        shuffle=False,
        dims=(320, 240),
        depth_scale=.25)
    model = Sequential([
        Conv2D(50, 3, activation='relu', padding='same', input_shape=train_seq.data_shape()),
        BatchNormalization(),
        Conv2D(50, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(50, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(80, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(80, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(100, 3, activation='relu', padding='same'),
        UpSampling2D(size=(2, 2)),
        BatchNormalization(),
        Conv2D(120, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(1, 3, activation='linear', padding='same'),
    ])

    H, W, _ = train_seq.data_shape()
    N = np.float32(W * H)

    model.compile(loss=loss.scale_invariant_gradient_loss(N), optimizer=Adam())
    return train_seq, val_seq, model
