import os
import numpy as np
from tensorflow.keras.layers import Input, Layer, Dense, SeparableConv2D, DepthwiseConv2D, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Add, Reshape, Activation, Lambda, GlobalAveragePooling2D
from tensorflow.keras import Model, utils
from tensorflow.keras.utils import plot_model
from batch_deform import BatchRetarget
from normalize import Normalize
from saliency import SpectralSaliency, MergeSaliency
import math

BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
}


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride,
                          name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_0_bn')(shortcut)
    else:
        shortcut = x


    x = Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = Conv2D(filters, kernel_size, padding='SAME',
               name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """''
    # if name == 'conv3':
    #     k3 = 3
    #     stride3 = get_conv_stride_size(x, 31, k3)
    #     # x_salient_nw_3 = salient_fn(x, k3, stride3, '3')
    #     # x_salient_nw_3_norm = Normalize()(x_salient_nw_3)
    #     x_salient_spectral_3 = SpectralSaliency()(x)
    #     x_salient_spectral_3 = SeparableConv2D(1, k3, stride3, use_bias = True, padding = 'VALID', activation = 'relu')(x_salient_spectral_3)
    #     x_salient_spectral_norm_3 = Normalize()(x_salient_spectral_3)
    #     # x_salient_merged_3 = MergeSaliency()([x_salient_nw_3_norm, x_salient_spectral_norm_3])
    #     x = BatchRetarget(name = 'retarget_3')([x, x_salient_spectral_norm_3])
    # if name == 'conv4':
    #     x_salient_nw_4 = salient_fn(x, '4')
    #     x_salient_nw_4_norm = Normalize()(x_salient_nw_4)
    #     x_salient_spectral_4 = SpectralSaliency()(x)
    #     x_salient_spectral_norm_4 = Normalize()(x_salient_spectral_4)
    #     x_salient_spectral_norm_4 = DepthwiseConv2D(3,padding = 'same', depth_multiplier = x_salient_spectral_norm_4.shape[3])(x_salient_spectral_norm_4)
    #     x_salient_merged_4 = MergeSaliency()([x_salient_nw_4_norm, x_salient_spectral_norm_4])
    #     x = BatchRetarget(name = 'retarget_4')([x, x_salient_merged_4])
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

'''
stack_fn: a function that returns output tensor for the
    stacked residual blocks.
'''
def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 8, stride1=2,name='conv3')#2
    x = stack1(x, 256, 36, stride1=2, name='conv4')#2
    x = stack1(x, 512, 3, stride1=2, name='conv5')
    return x

def get_conv_stride_size(x, n_out, k):
    n_in = x.shape[1]
    stride =  (n_in - k) / (n_out - 1)
    stride = math.floor(stride)
    return stride

def salient_fn(x, k, stride, block_index):
    n_in = x.shape[1]
    n_out = 31
    # for i in range(1):
    #     x = Conv2D(x.shape[3], 3, strides=1, use_bias=True, name = 'saliency_conv_' + str(i) + '_' + block_index, padding = 'SAME')(x)
    #     x = Activation('relu', name = 'saliency_conv_' + str(i) + '_' + block_index + '_relu')(x)
    x = Conv2D(1, k, strides = stride, name = 'saliency' + block_index, use_bias=False, padding = 'VALID', activation = 'relu')(x)
    return x


def ResNet152(
        use_bias=True,
        model_name='resnet152',
        weights=None,
        input_shape=(224, 224, 3),
        pooling='avg',
        classes=1000,
        pth_hist=None
    ):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: classes to classify images

    # Returns
        A Keras model instance.
    """

    bn_axis = 3
    img_input = Input(shape=input_shape)
    # x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv_deform1_pad')(img_input)
    k1 = 3
    stride1 = 1
    # x_salient_nw_1 = salient_fn(img_input, k1, stride1, '1')
    # x_salient_nw_1_norm = Normalize()(x_salient_nw_1)
    x_salient_spectral_1 = SpectralSaliency()(img_input)
    x_salient_spectral_1 = SeparableConv2D(1, k1, stride1, use_bias = False, padding = 'VALID', activation = None)(x_salient_spectral_1)
    x_salient_spectral_norm_1 = Normalize()(x_salient_spectral_1)
    # x_salient_merged_0 = MergeSaliency()([x_salient_nw_0_norm, x_salient_spectral_norm_0])
    x = BatchRetarget(name = 'retarget_0')([img_input, x_salient_spectral_norm_1])
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name='conv1_bn')(x)
    # k2 = 3
    # stride2 = 1
    # print(stride2)
    # x_salient_nw_2 = salient_fn(x, k2, stride2, '2')
    # x_salient_nw_2_norm = Normalize()(x_salient_nw_2)
    # x_salient_spectral_2 = SpectralSaliency()(x)
    # x_salient_spectral_2 = SeparableConv2D(x.shape[3], k2, stride2, use_bias = True, padding = 'VALID', activation = 'relu')(x_salient_spectral_2)
    # x_salient_spectral_2 = SeparableConv2D(x.shape[3], k2, stride2, use_bias = True, padding = 'VALID', activation = 'relu')(x_salient_spectral_2)
    # x_salient_spectral_2 = SeparableConv2D(x.shape[3], k2, stride2, use_bias = True, padding = 'VALID', activation = 'relu')(x_salient_spectral_2)
    # x_salient_spectral_2 = SeparableConv2D(1, k2, stride2, use_bias = True, padding = 'VALID')(x_salient_spectral_2)
    # x_salient_spectral_norm_2 = Normalize()(x_salient_spectral_2)
    # x_salient_merged_2 = MergeSaliency()([x_salient_nw_2_norm, x_salient_spectral_norm_2])
    # x = BatchRetarget(name = 'retarget_2')([x, x_salient_spectral_norm_2])
    # x = Activation('relu', name='conv1_relu')(x)
    # x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    # x = MaxPooling2D(name = 'max_pooling_deform_2', pool_size = (2, 2), strides = None, padding = 'valid')(x)
    x = stack_fn(x)
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs = img_input, outputs = x, name=model_name)

    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
            plot_model(model, to_file = os.path.join(pth_hist,'model.png'), dpi = 300)
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
    return model
