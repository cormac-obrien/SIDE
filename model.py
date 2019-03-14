import json
import os
import tarfile
import tempfile

from keras.models import model_from_json

def load_model(path):
    # Keras requires the weights to be specified by path so we have to extract
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(path, 'r') as f:
            f.extractall(tmpdir)

            with open(os.path.join(tmpdir, 'config.json'), 'r') as cfg_file:
                config = json.load(cfg_file)
            with open(os.path.join(tmpdir, 'model.json'), 'r') as model_file:
                model = model_from_json(json.load(model_file))
            model.load_weights(os.path.join(tmpdir, 'weights.h5'))

    return model, config
