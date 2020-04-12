'''
This module contains the custom model layer that can retarget feature maps
'''
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

class Retarget(Layer):
    '''
    Custom layer that retargets the innermost 2 dimensions of eature maps
    '''

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
        feature_maps, saliency_maps = maps
        #Add 1e-18 for edge case when activation maps is all 0's
        saliency_maps = tf.math.add(saliency_maps, 1e-18)
        x_grids, y_grids = create_grid_4d(saliency_maps)

        x_grids = tf.image.resize(
            x_grids,
            [feature_maps.shape[1], feature_maps.shape[2]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        y_grids = tf.image.resize(
            y_grids,
            [feature_maps.shape[1], feature_maps.shape[2]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        reatarget_maps = nearest_neighbour_interpolation(feature_maps, x_grids, y_grids)

        return reatarget_maps


def make_gaussian(size, fwhm=3, center=None):
    """
    Description
    -----------
    Makes a gaussian kernel

    Args
    ----
    size: size of desired kernel

    Returns
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

def create_grid_4d(x):
    '''
    Description
    -----------
    Read: https://github.com/recasens/Saliency-Sampler/blob/master/saliency_sampler.py
    However, instead of single image it creates grids on each channel of each batch in a feature map
    Input
    -----
    - x: tensor of shape (B, H, W, C)
    Returns
    -------
    - x_grids: tensor of shape (B, H, W, C)
    - y_grids: tensor of shape (B, H, W, C)
    '''
    # this is a hyperparameter. Increasing it increases compute cost but lowers the zoom
    grid_size = x.shape[1]*4
    # keep padding size low to keep area around zoom low
    padding_size = 3
    global_size = grid_size + 2 * padding_size
    num_channels = x.shape[3]

    P_basis_0 = tf.range(0, global_size, 1, dtype=tf.float32)
    P_basis_0 = tf.tile(P_basis_0, [global_size])
    P_basis_0 = tf.reshape(P_basis_0, [global_size, global_size])
    P_basis_0 = tf.math.subtract(P_basis_0, padding_size)
    P_basis_0 = tf.math.divide(P_basis_0, (grid_size-1.0))

    P_basis_0 = tf.expand_dims(P_basis_0, 0)
    P_basis_0 = tf.stack([P_basis_0]*num_channels, axis=3)

    P_basis_1 = tf.range(0, global_size, 1, dtype=tf.float32)
    P_basis_1 = tf.reshape(P_basis_1, (global_size, 1))
    P_basis_1 = tf.tile(P_basis_1, (1, global_size))
    P_basis_1 = tf.reshape(P_basis_1, [global_size, global_size])
    P_basis_1 = tf.math.subtract(P_basis_1, padding_size)
    P_basis_1 = tf.math.divide(P_basis_1, (grid_size-1.0))

    P_basis_1 = tf.expand_dims(P_basis_1, 0)
    P_basis_1 = tf.stack([P_basis_1]*num_channels, axis=3)

    gaussian_weight = make_gaussian(2*padding_size+1, 13)
    x = tf.image.resize(x, [grid_size, grid_size])
    x = tf.pad(x, tf.convert_to_tensor([[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], tf.int32), "REFLECT")

    gaussian_weights = tf.stack([gaussian_weight]*num_channels, axis=2)
    gaussian_weights = tf.stack([gaussian_weights]*1, axis=3)

    p_filter = tf.nn.depthwise_conv2d(
        x,
        gaussian_weights,
        strides=[1, 1, 1, 1],
        padding='VALID',
        data_format='NHWC'
    )
    x_mul_x = tf.math.multiply(P_basis_0, x)
    x_mul_x = tf.nn.depthwise_conv2d(
        x_mul_x,
        gaussian_weights,
        strides=[1, 1, 1, 1],
        padding='VALID',
        data_format='NHWC'
    )

    x_mul_y = tf.math.multiply(P_basis_1, x)
    x_mul_y = tf.nn.depthwise_conv2d(
        x_mul_y,
        gaussian_weights,
        strides=[1, 1, 1, 1],
        padding='VALID',
        data_format='NHWC'
    )

    x_filter = tf.math.divide_no_nan(x_mul_x, p_filter)
    x_grids = tf.math.subtract(tf.math.scalar_mul(2, x_filter), 1)
    x_grids = tf.clip_by_value(x_grids, clip_value_min=-1, clip_value_max=1)

    y_filter = tf.math.divide_no_nan(x_mul_y, p_filter)
    y_grids = tf.math.subtract(tf.math.scalar_mul(2, y_filter), 1)
    y_grids = tf.clip_by_value(y_grids, clip_value_min=-1, clip_value_max=1)

    return x_grids, y_grids

def get_pixel_value(feature_maps, x, y):
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

def nearest_neighbour_interpolation(feature_maps, x, y):
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
    Ia = get_pixel_value(feature_maps, x0, y0)
    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    #This is the trick. Assume all the neighbours are nearest neighbour
    out = tf.add_n([wa*Ia, wb*Ia, wc*Ia, wd*Ia])
    return out
