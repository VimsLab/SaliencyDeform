import tensorflow as tf
from tensorflow.keras.layers import Layer

class Normalize(Layer):

    def call(self, activation_maps):
        # print(activation_maps.shape)
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
        activation_maps = tf.math.subtract(1.0, activation_maps)

        # print(activation_maps.shape)
        # import pdb; pdb.set_trace()
        return activation_maps
