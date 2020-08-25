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

BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))

    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             batch_size=16,
             att_type=None,
             **kwargs):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: optional pooling mode for feature extraction
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

    if input_shape != (224,224,3):
        raise ValueError('Image dimesions need to be of the size 224 x 224')

    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
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

    x = transition_block(x, 0.5, name='pool2')
    if att_type == 'Retarget':
        ##################################################################################################
        x_attention1 = layers.DepthwiseConv2D(kernel_size=5,
                                              strides=(1, 1),
                                              padding='same')(x)
        x_attention2 = layers.Conv2D(filters=1,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding='same')(x)
        x_attention = WeightedAdd()(x_attention1, x_attention2)
        x_attention = layers.Activation('softmax')(x_attention)
        # x_attention = Normalize()(x_attention)
        x = Retarget()([x, x_attention])
        #################################################################################################
    elif att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type ==  'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    x = dense_block(x, blocks[1], name='conv3')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
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

    x = transition_block(x, 0.5, name='pool3')
    if att_type == 'Retarget':
        ###############################################################################################
        x_attention1 = layers.DepthwiseConv2D(kernel_size=5,
                                              strides=(1, 1),
                                              padding='same')(x)
        x_attention2 = layers.Conv2D(filters=1,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding='same')(x)
        x_attention = WeightedAdd()(x_attention1, x_attention2)
        x_attention = layers.Activation('softmax')(x_attention)
        # x_attention = Normalize()(x_attention)
        x = Retarget()([x, x_attention])
        ##############################################################################################
    elif att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    x = dense_block(x, blocks[2], name='conv4')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
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

    x = transition_block(x, 0.5, name='pool4')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    x = dense_block(x, blocks[3], name='conv5')
    if att_type == 'SE':
        U = layers.Conv2D(filters=int(backend.int_shape(x)[bn_axis]),
                          kernel_size=1,
                          strides=(1, 1),
                          padding='same')(x)
        x_attention = SELayer(reduction_ratio=16)(U)
        x = layers.multiply([x, x_attention])
    elif att_type == 'CBAM':
        x = CBAMLayer(reduction_ratio=16, kernel_size=7)(x)

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    x = layers.Dense(classes, activation='softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='30ee3e1110167f948a6b9946edeeb738')
            elif blocks == [6, 12, 32, 32]:
                weights_path = utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                batch_size=16,
                **kwargs):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, batch_size,
                    **kwargs)


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                batch_size=16,
                **kwargs):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, batch_size,
                    **kwargs)


def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                batch_size=16,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, batch_size,
                    **kwargs)
