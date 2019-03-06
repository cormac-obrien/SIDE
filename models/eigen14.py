from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape
from tensorflow.keras.models import Sequential

def build_model(input_shape, loss, optimizer):
    model = Sequential([
        Conv2D(96, 11, strides=4, activation='relu', padding='valid', input_shape=input_shape, name='conv1'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1'),
        BatchNormalization(),
        Conv2D(256, 5, activation='relu', padding='same', name='conv2'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2'),
        BatchNormalization(),
        Conv2D(384, 3, activation='relu', padding='same', name='conv3'),
        Conv2D(384, 3, activation='relu', padding='same', name='conv4'),
        Conv2D(256, 3, activation='relu', padding='same', name='conv5'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4800, activation='linear'),
        Reshape((60, 80, 1)),
    ])

    model.compile(loss=loss, optimizer=optimizer)
    return model
