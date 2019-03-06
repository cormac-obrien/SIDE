import tensorflow as tf
import tensorflow.keras.backend as K

def sobel(x):
    filt = K.variable([
        [[[1,  1]], [[0,  2]], [[-1,  1]]],
        [[[2,  0]], [[0,  0]], [[-2,  0]]],
        [[[1, -1]], [[0, -2]], [[-1, -1]]]
    ])

    return tf.nn.depthwise_conv2d(x, filt, strides=(1, 1, 1, 1), padding='SAME')


# [eigen 2015]
def scale_invariant_gradient_loss(N):
    def inner(y_pred, y_true):
        D = K.log(y_pred)
        D_star = K.log(y_true)
        d = D - D_star

        term1 = (1 / N) * K.sum(K.pow(d, 2))
        term2 = -(1 / (2 * N ** 2)) * K.pow(K.sum(d), 2)
        grad = sobel(d)
        term3 = (1 / N) * K.sum(K.pow(grad[:, :, :, 0], 2) + K.pow(grad[:, :, :, 1], 2))

        return term1 + term2 + term3
    return inner

