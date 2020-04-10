import tensorflow as tf
from tensorflow.keras.layers import Layer


"""
Makes a gaussian kernel
-----
- size: size of desired kernel
Returns
-------
- e2: gaussian kernel (size, size)
"""

def makeGaussian(size, fwhm = 3, center=None):
    fwhm = tf.convert_to_tensor(fwhm, tf.float32)
    x = tf.range(0, size, 1, tf.float32)
    y = tf.expand_dims(x, axis = 1)
    if center is None:
        x0 = y0 = tf.convert_to_tensor(size // 2, tf.float32)
    else:
        x0 = center[0]
        y0 = center[1]
    a1 = tf.math.scalar_mul(tf.convert_to_tensor(-4, tf.float32), tf.math.log(tf.convert_to_tensor(2, tf.float32)))
    a2 = tf.math.square(tf.math.subtract(x,x0))
    a3 = tf.math.square(tf.math.subtract(y,y0))
    a4 = tf.math.square(fwhm)
    e1 = tf.math.divide((tf.math.add( a2, a3 )), a4)
    e2 = tf.math.scalar_mul(a1, e1)
    e2 = tf.math.exp(e2)
    return e2

class GausBlur(Layer):
    def call(self, activation_maps):
        # print(activation_maps.shape)
        gaussian_weight = makeGaussian(8, 13)
        gaussian_weights = tf.stack([gaussian_weight]*activation_maps.shape[3], axis=2)
        gaussian_weights = tf.stack([gaussian_weights]*1, axis=3)
        activation_maps = tf.nn.depthwise_conv2d(activation_maps, gaussian_weights, strides=[1, 1, 1, 1], padding='SAME', data_format = 'NHWC')
        # print(activation_maps.shape)
        # import pdb; pdb.set_trace()
        return activation_maps

class Normalize(Layer):

    def call(self, activation_maps):

        activation_maps = tf.math.divide_no_nan(
           tf.math.subtract(
              activation_maps,
              tf.math.reduce_min(activation_maps,[1,2], keepdims = True)
           ),
           tf.subtract(
              tf.math.reduce_max(activation_maps,[1,2], keepdims = True),
              tf.math.reduce_min(activation_maps,[1,2], keepdims = True)
           )
        )

        # ck1 = tf.math.less_equal(activation_maps, tf.constant([0.5]))
        # activation_maps = tf.where(ck1, tf.zeros([activation_maps.shape[0],activation_maps.shape[1],activation_maps.shape[2],activation_maps.shape[3]]), activation_maps)
        return activation_maps


class Invert(Layer):
    '''
    Invert feature maps
    Feature maps must be between 0 and 1
    '''
    def call(self, activation_maps):
        # print(activation_maps.shape)
        activation_maps = tf.math.subtract(1.0, activation_maps)
        # print(activation_maps.shape)
        # import pdb; pdb.set_trace()
        return activation_maps
