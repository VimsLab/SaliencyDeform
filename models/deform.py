import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

class Deform(Layer):
    # def build(self, input_shape):
    #     self.degree = self.add_weight(shape=(input_shape[0] * input_shape[3],), initializer = 'Zeros', trainable = True, name='w')

    def call(self, activation_maps):
        # cos_pos = tf.math.cos(self.degree)
        # sin_pos = tf.math.sin(self.degree)
        # sin_neg = -1 * sin_pos
        # zeros = tf.zeros_like(cos_pos)
        # row1 = tf.stack([cos_pos, sin_neg, zeros], 0)
        # row2 = tf.stack([sin_pos, cos_pos, zeros], 0)
        # row1 = tf.transpose(row1, [1,0])
        # row2 = tf.transpose(row2, [1,0])
        # theta = tf.concat([row1, row2], 1)
        # theta = tf.reshape(theta, [activation_maps.shape[0] * activation_maps.shape[3],2,3])
        #
        # batch_grids_x, batch_grids_y = affine_grid_generator_4d(activation_maps, theta)
        # activation_maps = bilinear_sampler_4d(activation_maps, batch_grids_x, batch_grids_y)
        saliency_maps = compute_spectral_saliency_4d(activation_maps)
        x_grids, y_grids = create_grid_4d(saliency_maps)
        activation_maps_out = bilinear_sampler_4d(activation_maps, x_grids, y_grids)
        #
        # print(activation_maps.shape, activation_maps_out.shape, '\n')
        # tf.print(activation_maps.shape, activation_maps_out.shape, '\n')

        return activation_maps_out

"""
Convolve input tesnor with a box filter of shape (3,3)
-----
- img: feature map of shape (B,C,H,W), where H is 64 and W is H/2
Returns
-------
- out: convolved feature map of shape (B,C,H,W)
"""
def box_filter_4d(img):
    filter = tf.constant([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], tf.float32)
    filters = tf.stack([filter]*img.shape[3], axis = 2)
    filters = tf.stack([filters]*img.shape[3], axis = 2)
    out = tf.nn.conv2d(img, filters, strides=[1, 1, 1, 1], padding='SAME')
    return out

'''
Spectral saliency is computed according to https://ieeexplore.ieee.org/abstract/document/4270292
on on each channel of each batch in a feature map. Gaussian blur is not applied at the end
Input
-----
- x: tensor of shape (B, H, W, C)
Returns
-------
- mag: saliency map of shape (B, H, W, C)
'''
def compute_spectral_saliency_4d(activation_map_inp):
    # activation map is resized according to the original paper
    activation_map = tf.image.resize(activation_map_inp , [64,64])
    # change activation map format from [B,H,W,C] to [B,C,H,W] since rfft2d works on the innermost 2 dimensions ( Read documentation)
    # tf.print(activation_map)
    activation_map = tf.transpose(activation_map, [0,3,1,2])
    fft_activation_map = tf.signal.rfft2d(activation_map)
    rfft_activation_map = tf.math.real(fft_activation_map)
    ifft_activation_map = tf.math.imag(fft_activation_map)
    mag = tf.math.sqrt(tf.math.add(tf.math.square(rfft_activation_map), tf.math.square(ifft_activation_map)))
    ck1 = tf.math.equal(mag, tf.constant([0.0]))
    mag = tf.where(ck1, tf.ones([1,mag.shape[1],mag.shape[2],mag.shape[3]]), mag)
    log_mag = tf.math.log(mag)
    box_filter = box_filter_4d(log_mag)
    spectralResidual = tf.math.subtract(log_mag, box_filter)
    rfft_activation_map = tf.math.multiply(rfft_activation_map, tf.math.divide_no_nan(spectralResidual, mag))
    ifft_activation_map = tf.math.multiply(ifft_activation_map, tf.math.divide_no_nan(spectralResidual, mag))
    c = tf.dtypes.complex(rfft_activation_map, ifft_activation_map)
    c = tf.signal.ifft2d(c)
    c_real = tf.math.real(c)
    c_imag = tf.math.imag(c)
    mag = tf.math.add(tf.math.square(c_real), tf.math.square(c_imag))

    mag = tf.transpose(mag, [0,2,3,1])
    mag = tf.image.resize(mag , [activation_map_inp.shape[1],activation_map_inp.shape[2]])
    mag = tf.math.divide_no_nan(
       tf.math.subtract(
          mag,
          tf.math.reduce_min(mag)
       ),
       tf.subtract(
          tf.math.reduce_max(mag),
          tf.math.reduce_min(mag)
       )
    )
    # gaussian_weight = makeGaussian(5, 13)
    # gaussian_weights = tf.stack([gaussian_weight]*mag.shape[3], axis = 2)
    # gaussian_weights = tf.stack([gaussian_weights]*1, axis = 3)
    # mag = tf.nn.depthwise_conv2d(mag, gaussian_weights, strides=[1, 1, 1, 1], padding='VALID', data_format = 'NHWC')
    return mag

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
    grid_size = 31
    padding_size =  30
    global_size = grid_size + 2 * padding_size
    #for multigpu divide batch size by num_gpu
    batch_size = x.shape[0]
    #make size equal to max pooling
    # out_map_size = tf.cast(tf.math.divide(x.shape[1],2), tf.int32)
    out_map_size = x.shape[1]
    P_basis_0 = tf.range(0, global_size, 1, dtype = tf.float32)
    P_basis_0 = tf.tile(P_basis_0,[global_size])
    P_basis_0 = tf.reshape(P_basis_0,[global_size, global_size])
    P_basis_0 = tf.math.subtract(P_basis_0, padding_size)
    P_basis_0 = tf.math.divide_no_nan(P_basis_0, (grid_size-1.0))

    P_basis_0 = tf.stack([P_basis_0]*batch_size, axis = 0)
    P_basis_0 = tf.stack([P_basis_0]*x.shape[3], axis = 3)

    P_basis_1 = tf.range(0, global_size, 1, dtype = tf.float32)
    P_basis_1 = tf.reshape(P_basis_1, (global_size, 1))
    P_basis_1 = tf.tile(P_basis_1, (1, global_size))
    P_basis_1 = tf.reshape(P_basis_1,[global_size, global_size])

    P_basis_1 = tf.math.subtract(P_basis_1, padding_size)
    P_basis_1 = tf.math.divide_no_nan(P_basis_1, (grid_size-1.0))

    P_basis_1 = tf.stack([P_basis_1]*batch_size, axis = 0)
    P_basis_1 = tf.stack([P_basis_1]*x.shape[3], axis = 3)

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

    x_grids = tf.image.resize(x_grids, [out_map_size,out_map_size] , method= tf.image.ResizeMethod.BILINEAR)
    y_grids = tf.image.resize(y_grids, [out_map_size,out_map_size] , method= tf.image.ResizeMethod.BILINEAR)

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
    batch_size = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]
    num_channels = x.shape[3]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))

    b = tf.tile(batch_idx, (1, height, width))
    b = tf.stack([b]*num_channels, axis = 3)

    num_channels_idx = tf.range(0, num_channels)
    num_channels_idx = tf.reshape(num_channels_idx, (num_channels, 1, 1))

    c = tf.tile(num_channels_idx, (1, height, width))
    c = tf.transpose(c, [1,2,0])
    c = tf.stack([c]*batch_size, axis = 0)

    indices = tf.stack([b, y, x, c], 4)
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

    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

"""
This function returns a sampling grid, which when
used with the bilinear sampler on the input feature
map, will create an output feature map that is an
affine transformation [1] of the input feature map.
Input
-----
- a_map: feature maps of images (B, H, W, C)
- theta: affine transform matrices of shape (num_batch * num_classes, 2, 3).
  For each image in the batch, we have 6 theta parameters of
  the form (2x3) that define the affine transformation T.
Returns
-------
- batch_grids_x: grid of shape (B, H, W, C).
- batch_grids_y: grid of shape (B, H, W, C).
"""
def affine_grid_generator_4d(a_map, theta):
    num_batch = a_map.shape[0]
    num_channels = a_map.shape[3]
    height = a_map.shape[1]
    width = a_map.shape[2]

    a_map = tf.transpose(a_map, [0,3,1,2])
    a_map = tf.reshape(a_map, [num_batch * num_channels, height, width] )

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, height)
    y = tf.linspace(-1.0, 1.0, width)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)

    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    sampling_grid = tf.stack([sampling_grid] * num_batch * num_channels, 0)

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)

    batch_grids_x = batch_grids[:,0,:]
    batch_grids_y = batch_grids[:,1,:]

    batch_grids_x = tf.reshape(batch_grids_x, [num_batch, num_channels, height, width])
    batch_grids_x = tf.transpose(batch_grids_x, [0, 2, 3, 1])

    batch_grids_y = tf.reshape(batch_grids_y, [num_batch, num_channels, height, width])
    batch_grids_y = tf.transpose(batch_grids_y, [0, 2, 3, 1])


    return batch_grids_x, batch_grids_y
