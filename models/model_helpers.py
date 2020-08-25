import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import MinMaxNorm

def make_gaussian(
        size=None,
        fwhm=3,
        center=None
    ):
    """
    Description
    -----------
    Makes a gaussian kernel

    Args
    ----
    size: size of desired kernel

    Returns:
    -------
    e2: gaussian kernel (size, size)
    """
    fwhm = tf.convert_to_tensor(fwhm, tf.float32)
    x = tf.range(0, size, 1, tf.float32)
    y = tf.expand_dims(x, axis=1)
    if center is None:
        x0 = y0 = tf.convert_to_tensor(size // 2, tf.float32)
    else:
        x0 = center[0]
        y0 = center[1]
    a1 = tf.math.scalar_mul(tf.convert_to_tensor(-4, tf.float32), tf.math.log(tf.convert_to_tensor(2, tf.float32)))
    a2 = tf.math.square(tf.math.subtract(x, x0))
    a3 = tf.math.square(tf.math.subtract(y, y0))
    a4 = tf.math.square(fwhm)
    e1 = tf.math.divide((tf.math.add(a2, a3)), a4)
    e2 = tf.math.scalar_mul(a1, e1)
    e2 = tf.math.exp(e2)
    return e2

class GausBlur(Layer):
    """
    Description
    -----------
    Layer to perform gaussian blue on the innermost 2 dimesions of feature maps
    Blur is performed seprately done on each channel of a batch
    """

    def call(self, feature_maps):
        """
        Description
        -----------
        Perform depthwise gaussian blur on the each channel of feature map

        Args
        ----
        feature_maps: feature_maps of shape (B, H, W, C)

        Returns
        -------
        blur_maps: gaussian blurred feature_maps
        """
        gaussian_weight = make_gaussian(5, 13)
        gaussian_weights = tf.stack([gaussian_weight]*feature_maps.shape[3], axis=2)
        gaussian_weights = tf.stack([gaussian_weights]*1, axis=3)
        blur_maps = tf.nn.depthwise_conv2d(input=feature_maps,
                                           filter=gaussian_weights,
                                           strides=[1, 1, 1, 1],
                                           padding='SAME',
                                           data_format='NHWC'
                                          )
        return blur_maps

class Normalize(Layer):
    """
    Description
    -----------
    Layer to normalize the values of the innermost 2 dimesions of feature maps
    of shape (B, H, W, C) between 0 and 1
    """

    def call(self, feature_maps):
        """
        Description
        -----------
        Normalize the values of the innermost 2 dimesions of each channel of a batch of feature_maps
        between 0 to 1

        Args
        ----
        feature_maps: feature_maps of shape (B, H, W, C)

        Returns
        -------
        normalized_maps: normalized feature_maps
        """
        normalized_maps = tf.math.divide_no_nan(
            tf.math.subtract(
                feature_maps,
                tf.math.reduce_min(feature_maps, [1, 2], keepdims=True)
            ),
            tf.subtract(
                tf.math.reduce_max(feature_maps, [1, 2], keepdims=True),
                tf.math.reduce_min(feature_maps, [1, 2], keepdims=True)
            )
        )
        return normalized_maps


class Invert(Layer):
    """
    Description
    -----------
    Layer to invert the innermost 2 dimesions of feature maps
    """

    def call(self, feature_maps):
        '''
        Description
        -----------
        Invert value of feature maps.
        Innermost 2 dimesions of activation_maps must be in the range of 0 to 1.

        Args:
        ----
        feature_maps: feature maps of shape (B,H,W,C)

        Returns:
        -------
        invert_maps: subtracted innermost 2 dimesions of feature maps from 1
        '''
        invert_maps = tf.math.subtract(1.0, feature_maps)
        return invert_maps



class WeightedAdd(Layer):
    """
    Description
    -----------
    Layer to invert the innermost 2 dimesions of feature maps
    """

    def __init__(self, name=None):
        super(WeightedAdd, self).__init__()

    def build(self, shape):
        weight_constraint = MinMaxNorm(min_value=0.0,
                                       max_value=1.0,
                                       rate=1.0,
                                       axis=0)
        self.w1 = self.add_weight(name='w1',
                                  shape=(shape[3],),
                                  initializer="ones",
                                  trainable=True,
                                  constraint=weight_constraint)
        # self.w2 = self.add_weight(name='w2', shape=(1,), initializer="ones", trainable=True)
        self.w2 = 1 - self.w1

    def call(self, feature_maps1, feature_maps2):
        '''
        Description
        -----------
        Invert value of feature maps.
        Innermost 2 dimesions of activation_maps must be in the range of 0 to 1.

        Args:
        ----
        feature_maps: feature maps of shape (B,H,W,C)

        Returns:
        -------
        invert_maps: subtracted innermost 2 dimesions of feature maps from 1
        '''
        add_maps = self.w1*feature_maps1 + self.w2*feature_maps2
        return add_maps
