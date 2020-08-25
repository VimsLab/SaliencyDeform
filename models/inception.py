import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras import backend
from tensorflow.keras.utils import plot_model
#Custom Layers
from retarget import Retarget
from squeeze_excite import SELayer
from BAM import BAMLayer
from CBAM import CBAMLayer

from model_helpers import Normalize, Invert, GausBlur, WeightedAdd
# backend.set_image_data_format('channels_first')

WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

layer_name_count = {}

def assign_name(layer_name):
    # layer_name = layer_name[:layer_name.rindex('/')]
    if layer_name not in layer_name_count:
        layer_name_count[layer_name] = 0
        return layer_name + '_inception'
    else:
        layer_name_count[layer_name] += 1
        layer_name = layer_name + '_' + str(layer_name_count[layer_name]) + '_inception'
        return layer_name

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    bn_axis = 3
    name = assign_name('conv2d')
    x = layers.Conv2D(filters,
                      (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=False,
                      name=name)(x)
    name = assign_name('batch_normalization')
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=name)(x)
    name = assign_name('activation')
    x = layers.Activation('relu', name=name)(x)
    return x

def InceptionV3(include_top=True,
                weights='imagenet',
                input_shape=None,
                pooling=None,
                classes=1000,
                batch_size=16,
                by_name=False,
                att_type='',
                **kwargs):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if att_type not in ['baseline', 'SE', 'BAM', 'CBAM', 'Retarget']:
        raise ValueError('Custom Attention Module of required type is required to train'
                         'custom models')

    if input_shape != (299,299,3):
        raise ValueError('Image dimesions need to be of the size 299 x 299')

    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')

    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    name = assign_name('max_pooling2d')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),name=name)(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    if att_type == 'BAM':
        x_attention = BAMLayer(reduction_ratio=16, dilation_val=4)(x)
        x_attention = layers.Activation('sigmoid')(x_attention)
        x_attention = tf.math.add(1.0, x_attention)
        x_attention = layers.multiply([x, x_attention])
        x = layers.Add()([x, x_attention])
    name = assign_name('max_pooling2d')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),name=name)(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    name = assign_name('average_pooling2d')
    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          name=name)(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    name = assign_name('average_pooling2d')
    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          name=name)(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)
    elif att_type == 'Retarget':
        ###############################################CUSTOM CODE#####################################
        x_attention1 = layers.DepthwiseConv2D(kernel_size=5,
                                              strides=(1, 1),
                                              padding='same')(x)
        x_attention2 = layers.Conv2D(filters=1,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding='same')(x)
        x_attention = WeightedAdd()(x_attention1, x_attention2)
        # x_attention = layers.Activation('softmax')(x_attention)
        x_attention = Normalize()(x_attention)
        x = Retarget()([x, x_attention])
        ###############################################CUSTOM CODE#####################################

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    name = assign_name('average_pooling2d')
    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          name=name)(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)
    elif att_type == 'BAM':
        x_attention = BAMLayer(reduction_ratio=16, dilation_val=4)(x)
        x_attention = layers.Activation('sigmoid')(x_attention)
        x_attention = tf.math.add(1.0, x_attention)
        x_attention = layers.multiply([x, x_attention])
        x = layers.Add()([x, x_attention])

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    name = assign_name('max_pooling2d')
    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), name=name)(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')
    if att_type == 'Retarget':
        ###############################################CUSTOM CODE#####################################
        x_attention1 = layers.DepthwiseConv2D(kernel_size=5,
                                              strides=(1, 1),
                                              padding='same')(x)
        x_attention2 = layers.Conv2D(filters=1,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding='same')(x)
        x_attention = WeightedAdd()(x_attention1, x_attention2)
        # x_attention = layers.Activation('softmax')(x_attention)
        x_attention = Normalize()(x_attention)
        x = Retarget()([x, x_attention])
        ###############################################CUSTOM CODE#####################################
    elif att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    name = assign_name('average_pooling2d')
    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          name=name)(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        name = assign_name('average_pooling2d')
        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name)(x)

        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))
        if att_type == 'SE':
            U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                              kernel_size=1,
                              strides=(1, 1),
                              padding='same')(x)
            x_attention = SELayer(reduction_ratio=16)(U)
            x = layers.multiply([x, x_attention])
        elif att_type == 'CBAM':
            x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    name = assign_name('average_pooling2d')
    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          name=name)(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)
    elif att_type == 'BAM':
        x_attention = BAMLayer(reduction_ratio=16, dilation_val=4)(x)
        x_attention = layers.Activation('sigmoid')(x_attention)
        x_attention = tf.math.add(1.0, x_attention)
        x_attention = layers.multiply([x, x_attention])
        x = layers.Add()([x, x_attention])


    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    name = assign_name('max_pooling2d')
    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), name=name)(x)

    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        name = assign_name('average_pooling2d')
        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same', name=name)(x)

        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
        if att_type == 'SE':
            U = layers.Conv2D(filters=int(backend.int_shape(x)[channel_axis]),
                              kernel_size=1,
                              strides=(1, 1),
                              padding='same')(x)
            x_attention = SELayer(reduction_ratio=16)(U)
            x = layers.multiply([x, x_attention])
        elif att_type == 'CBAM':
            x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(1000, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
            x = layers.Dense(classes, activation='softmax', name='dense')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
            x = layers.Dense(classes, activation='softmax', name='dense')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
            model.load_weights(weights_path, by_name=False)
        else:
            if not by_name:
                weights_path = utils.get_file(
                    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='bcbd6486424b2319ff4ef7d526e38f63')
                model.load_weights(weights_path, by_name=by_name)
            else:
                weights_path = '/home/jakep/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop_by_name3.h5'
                model.load_weights(weights_path, by_name=by_name)
    elif weights is not None:
        model.load_weights(weights)
    return model
