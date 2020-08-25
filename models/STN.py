'''
This module contains the custom model layer that can retarget feature maps
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import math

class STN(Layer):
    '''
    Custom layer that retargets the innermost 2 dimensions of eature maps
    '''

    def __init__(self):
        super(STN, self).__init__()

    def build(self, shape):


    def call(self, maps):
        '''
        Description
        -----------
        Retargets the innermost two dimensions of feature maps according to their saliency maps

        Args
        ----
        maps: a list containing feature maps and saliency_maps both of shape (B, H, W, C)

        Returns
        -------
        reatarget_maps: retargeted feature maps
        '''

        # print(self.sigma)
        # tf.print(self.sigma)
        #Add 1e-18 for edge case when activation maps is all 0's
        # saliency_maps = tf.math.add(saliency_maps, 1e-18)


        x_grids, y_grids = self.create_grid_4d(saliency_maps)
        reatarget_maps = self.nearest_neighbour_interpolation(feature_maps, x_grids, y_grids)

        return reatarget_maps

    def get_pixel_value(self, feature_maps, x, y):
        """
        Given coordinate vectors x and y it gathers pixel values from 4D feature maps in NHWC format
        Input
        -----
        feature_maps: feature_maps image of tensor of shape (B, H, W, C)
        x: x coordinates of grid of shape (B, H, W, C)
        y: y coordinates of grid of shape (B, H, W, C)

        Returns
        -------
        output: tensor of shape (B, H, W, C)
        """
        batch_size = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]
        num_channels = x.shape[3]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))

        b = tf.tile(batch_idx, (1, height, width))
        b = tf.stack([b]*num_channels, axis=3)

        num_channels_idx = tf.range(0, num_channels)
        num_channels_idx = tf.reshape(num_channels_idx, (num_channels, 1, 1))

        c = tf.tile(num_channels_idx, (1, height, width))
        c = tf.transpose(c, [1, 2, 0])
        c = tf.stack([c]*batch_size, axis=0)

        indices = tf.stack([b, y, x, c], 4)
        output = tf.gather_nd(feature_maps, indices)

        return output

    def nearest_neighbour_interpolation(self, feature_maps, x, y):
        '''
        Performs differentiable nearest neighbour sampling of the feature maps according to the
        coordinates provided by the sampling grid. Note that
        the sampling is NOT done identically for each channel of the input.

        Input
        -----
        feature_maps: feature_maps of shape (B, H, W, C)
        x, y: output coordinates of grid

        Returns
        -------
        out: interpolated feature_maps

        '''
        H = tf.shape(feature_maps)[1]
        W = tf.shape(feature_maps)[2]

        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x + 0.5), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y + 0.5), 'int32')
        y1 = y0 + 1
        # # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        # get pixel value at nearest neighbour
        Ia = self.get_pixel_value(feature_maps, x0, y0)

        # # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')
        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)
        # #This is the trick. Assume all the neighbours are nearest neighbour
        out = tf.add_n([Ia*wa, Ia*wb, Ia*wc, Ia*wd])
        # out = Ia + x - x + y - y
        return out
