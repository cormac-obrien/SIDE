from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import applications

def build_vgg():
    # scale 1
    vgg = applications.VGG16(weights='imagenet', include_top=False)
    block5_pool = vgg.layers[-1]

    dense1 = Dense(block5_pool, activation='relu')
    drop1 = Dropout(.5)(dense1)

    dense2 = Dense(drop1, activation='relu')
    drop2 = Dropout(.5)(dense2)
