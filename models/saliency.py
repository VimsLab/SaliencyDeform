import tensorflow as tf
from tensorflow.keras.layers import Layer


class SpectralSaliency(Layer):
    def call(self, activation_maps):
        spectral_saliency = compute_spectral_saliency_4d(activation_maps)
        # print(spectral_saliency.shape)
        # channel_reduced_spectral_saliency = tf.math.reduce_sum(spectral_saliency, 3)
        # print(channel_reduced_spectral_saliency.shape)
        # channel_reduced_spectral_saliency = tf.expand_dims(channel_reduced_spectral_saliency, 3)
        # print(channel_reduced_spectral_saliency.shape)
        gaussian_weight = makeGaussian(8, 13)
        gaussian_weights = tf.stack([gaussian_weight]*spectral_saliency.shape[3], axis = 2)
        gaussian_weights = tf.stack([gaussian_weights]*1, axis = 3)
        spectral_saliency = tf.nn.depthwise_conv2d(spectral_saliency, gaussian_weights, strides=[1, 1, 1, 1], padding='SAME', data_format = 'NHWC')
        # import pdb; pdb.set_trace()
        return spectral_saliency


class AddChannels(Layer):
    def call(self, activation_maps):
        channel_saliency = tf.math.reduce_sum(activation_maps, 3)
        channel_saliency = tf.expand_dims(channel_saliency, 3)
        gaussian_weight = makeGaussian(8, 13)
        gaussian_weights = tf.stack([gaussian_weight]*channel_saliency.shape[3], axis=2)
        gaussian_weights = tf.stack([gaussian_weights]*1, axis=3)
        channel_saliency = tf.nn.depthwise_conv2d(
            channel_saliency,
            gaussian_weights,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        # print(channel_saliency.shape)
        # import pdb; pdb.set_trace()
        return channel_saliency


class MergeSaliency(Layer):
    def build(self, input_shape):
        self.nw_saliency_wt = self.add_weight(shape=(1,), initializer = 'Ones', trainable = True, name='nw_saliency_wt')
        self.spectral_saliency_wt = self.add_weight(shape=(1,), initializer = 'Ones', trainable = True, name='spectral_saliency_wt')

    def call(self, maps):
        # print(self.nw_saliency_wt.shape, self.spectral_saliency_wt.shape)
        nw_saliency, spectral_saliency = maps
        nw_saliency = tf.math.multiply(nw_saliency, self.nw_saliency_wt )
        spectral_saliency = tf.math.multiply(spectral_saliency, self.spectral_saliency_wt)

        # print(nw_saliency.shape, spectral_saliency.shape)
        # print(nw_saliency.shape, spectral_saliency.shape)
        merged_saliency = tf.add_n([spectral_saliency, nw_saliency])
        # print(merged_saliency.shape)
        # import pdb; pdb.set_trace()
        return merged_saliency



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
    activation_map = tf.image.resize(activation_map_inp , [activation_map_inp.shape[1],activation_map_inp.shape[2]])
    # change activation map format from [B,H,W,C] to [B,C,H,W] since rfft2d works on the innermost 2 dimensions ( Read documentation)
    # tf.print(activation_map)
    activation_map = tf.transpose(tf.cast(activation_map, tf.complex64),[0,3,1,2])
    fft_activation_map = tf.signal.fft2d(activation_map)
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
    return mag
