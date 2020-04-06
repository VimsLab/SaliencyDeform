import tensorflow as tf
from tensorflow.keras.layers import Layer


class BatchRetarget(Layer):

    def call(self, maps):
        activation_maps, saliency_maps = maps
        # print(activation_maps.shape, saliency_maps.shape)
        x_grids, y_grids = create_grid_4d(saliency_maps)
        # print(x_grids.shape, y_grids.shape)
        # import pdb; pdb.set_trace()
        x_grids = tf.image.resize(x_grids, [activation_maps.shape[1],activation_maps.shape[2]] , method= tf.image.ResizeMethod.BILINEAR)
        y_grids = tf.image.resize(y_grids, [activation_maps.shape[1],activation_maps.shape[2]] , method= tf.image.ResizeMethod.BILINEAR)
        x_grids = tf.squeeze(x_grids,3)
        y_grids = tf.squeeze(y_grids,3)
        # print(x_grids.shape, y_grids.shape)
        activation_maps_out = bilinear_sampler_4d(activation_maps, x_grids, y_grids)
        # print(activation_maps_out.shape)
        # import pdb; pdb.set_trace()
        return activation_maps_out

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

'''
A saliency based zoom grid is created as per https://github.com/recasens/Saliency-Sampler/blob/master/saliency_sampler.py
Instead of a single image it creates grids on each channel of each batch in a feature map
Input
-----
- x: tensor of shape (B, H, W, C)
Returns
-------
- x_grids: tensor of shape (B, H, W, C)
- y_grids: tensor of shape (B, H, W, C)
'''
def create_grid_4d(x):

    grid_size = x.shape[1]
    padding_size = 5
    global_size = grid_size + 2 * padding_size
    batch_size = tf.shape(x)[0]

    P_basis_0 = tf.range(0, global_size, 1, dtype = tf.float32)
    P_basis_0 = tf.tile(P_basis_0,[global_size])
    P_basis_0 = tf.reshape(P_basis_0,[global_size, global_size])
    P_basis_0 = tf.math.subtract(P_basis_0, padding_size)
    P_basis_0 = tf.math.divide_no_nan(P_basis_0, (grid_size-1.0))
    P_basis_0 = tf.stack([P_basis_0]*x.shape[3], axis = 2)

    P_basis_0 = tf.cast(P_basis_0, tf.float32)
    P_basis_1 = tf.range(0, global_size, 1, dtype = tf.float32)
    P_basis_1 = tf.reshape(P_basis_1, (global_size, 1))
    P_basis_1 = tf.tile(P_basis_1, (1, global_size))
    P_basis_1 = tf.reshape(P_basis_1,[global_size, global_size])
    P_basis_1 = tf.math.subtract(P_basis_1, padding_size)
    P_basis_1 = tf.math.divide_no_nan(P_basis_1, (grid_size-1.0))
    P_basis_1 = tf.stack([P_basis_1]*x.shape[3], axis = 2)

    gaussian_weight = makeGaussian(2*padding_size+1, 13)
    x = tf.image.resize(x , [grid_size,grid_size])
    x = tf.pad(x, tf.convert_to_tensor([[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]], tf.int32), "REFLECT")

    gaussian_weights = tf.stack([gaussian_weight]*x.shape[3], axis = 2)
    gaussian_weights = tf.stack([gaussian_weights]*x.shape[3], axis = 3)

    p_filter = tf.nn.conv2d(x, gaussian_weights, strides=[1, 1, 1, 1], padding='VALID', data_format = 'NHWC')

    x_mul_x = tf.math.multiply(P_basis_0,x)
    x_mul_x = tf.nn.conv2d(x_mul_x, gaussian_weights, strides=[1, 1, 1, 1], padding='VALID', data_format = 'NHWC')

    x_mul_y = tf.math.multiply(P_basis_1,x)
    x_mul_y = tf.nn.conv2d(x_mul_y, gaussian_weights, strides=[1, 1, 1, 1], padding='VALID', data_format = 'NHWC')

    x_filter = tf.math.divide_no_nan(x_mul_x, p_filter)
    x_grids = tf.math.subtract(tf.math.scalar_mul(2, x_filter),1)
    x_grids = tf.clip_by_value(x_grids,clip_value_min=-1,clip_value_max=1)

    y_filter = tf.math.divide_no_nan(x_mul_y, p_filter)
    y_grids = tf.math.subtract(tf.math.scalar_mul(2, y_filter),1)
    y_grids = tf.clip_by_value(y_grids,clip_value_min=-1,clip_value_max=1)

    return x_grids, y_grids

"""
Given coordinate vectors x and y it gathers pixel values from 4D feature maps in NHWC format
Input
-----
- img: tensor of shape (B, H, W, C)
- x: Tensor of shape (B, H, W, C)
- y: flattened tensor of shape (B, H, W, C)
Returns
-------
- output: tensor of shape (B, H, W, C)
"""
def get_pixel_value_4d(img, x, y):

    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))
    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

'''
Performs bilinear sampling of the input images according to the
coordinates provided by the sampling grid. Note that
the sampling is NOT done identically for each channel of the input.

Input
-----
- img: batch of images in (B, H, W, C) layout.
- x, y: output of an grid

Returns
-------
- out: interpolated images according to grids. Same shape as img.

'''
def bilinear_sampler_4d(img, x, y):

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    # get pixel value at corner coords
    Ia = get_pixel_value_4d(img, x0, y0)
    Ib = get_pixel_value_4d(img, x0, y1)
    Ic = get_pixel_value_4d(img, x1, y0)
    Id = get_pixel_value_4d(img, x1, y1)
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

    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out
