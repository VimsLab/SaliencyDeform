'''
Visual output of feature_maps from a forward pass
'''
import os
import sys
import yaml
pth_config = './config'
with open(os.path.join(pth_config, 'clef2016.yml'), 'r') as config_fl:
    config = yaml.load(config_fl)
pth_data = config['pth_data']
pth_utils = config['pth_utils']
pth_models = config['pth_models']
pth_weights = config['pth_weights']
pth_visual = config['pth_visual']
pths_import = [
    pth_data,
    pth_utils,
    pth_models,
    pth_weights,
]
for pth_import in pths_import:
    if pth_import not in sys.path:
        sys.path.append(pth_import)

from custom_common_densenet import DenseNet169

from load_data import image_data_generator
from preprocess import center_crop
from util_visualize import visualize_feature_maps

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.optimizers import Adam
if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Load and compile model. Model must be the same as the one used for training
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    classes = config['classes']
    batch_size = config['batch_size']
    run = '17124'
    # run = '29303'
    # run = '79092'

    weight_name = os.path.join(pth_weights,'{0}/cp-0030.ckpt'.format(run))
    model = DenseNet169(
        include_top=False,
        weights=None,
        input_shape=(nw_img_rows, nw_img_cols, 3),
        pooling='avg',
        classes=classes,
        batch_size=batch_size
    )
    # model.compile(optimizer=SGDW(lr=0.0001, weight_decay=1e-6, momentum=0.9),
    #               loss='categorical_crossentropy',
    #               metrics=['categorical_accuracy'])
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    #name of layers to be visualized
    layer_names = {
        'pool2_conv',
        # 'normalize_pool2',
        'multiply_1',
        'add_1',
        'retarget',
        'pool2_pool',

        'pool3_conv',
        'multiply_3',
        # 'add_3',
        # 'normalize_pool3',
        'retarget_1',
        'pool3_pool',

        'pool4_conv',
        # 'add_5',
        'multiply_5',
        # 'normalize_pool4',
        'retarget_2',
        'pool3_pool',

        # # 'conv2d_1',
        # 'conv2d_2'
        # 'conv2d'
        # 'conv2_block3_out',
        # 'conv1_pad',
        # 'depthwise_conv_0_conv1_pad',
        # 'depthwise_conv_1_conv1_pad',
        # 'depthwise_conv_2_conv1_pad',
        # 'normalize_conv1_pad',
        # 'retarget_conv1_pad',
        # 'depthwise_conv_0_conv3',
        # 'normalize_conv3',
        # 'retarget_conv3',
        # 'conv4_block36_out',
        # 'normalize_conv5',
        # 'retarget_conv5',
        # 'depthwise_conv_0_conv5',
        # 'conv3_block8_out',
        # 'depthwise_conv_0_conv4',
        # 'normalize_conv4',
        # 'retarget_conv4',
        # # 'conv5_block3_out',
        # 'depthwise_conv_0_avg_pool',
        # 'normalize_avg_pool',
        # 'retarget_avg_pool'
    }
    visualize_layers_at = []
    for index, layer in enumerate(model.layers):
        if layer.name in layer_names:
            visualize_layers_at.append(index)
    for visualize_layer_at in visualize_layers_at:
        #Load data for visualization
        ip_img_cols = config['ip_img_cols']
        ip_img_rows = config['ip_img_rows']
        VISUALIZE = os.path.join(pth_data, 'visualize')
        visual_gen = image_data_generator(
            in_dir=VISUALIZE,
            preprocessing_function=preprocess_input_densenet,
            target_size=(ip_img_rows, ip_img_cols),
            batch_size=batch_size,
            horizontal_flip=False,
            shuffle=False
        )
        steps = len(visual_gen)
        visual_gen = center_crop(visual_gen, ip_img_rows, ip_img_cols, nw_img_cols, batch_size)
        layer_name = model.layers[visualize_layer_at].name
        model_partial = Model(inputs=model.input, outputs=model.layers[visualize_layer_at].output)
        print('Partial model compiled with its last layer {0} compiled'.format(layer_name))
        feature_maps = model_partial.predict(visual_gen, steps=steps)
        print('Predicted')
        channels = 4
        # import numpy as np
        feature_maps = feature_maps[0:10,:,:,:]
        num_img = feature_maps.shape[0]
        VISUAL = os.path.join(pth_visual, run)
        visualize_feature_maps(
            num_img=num_img,
            layer_index=visualize_layer_at,
            layer_name=layer_name,
            channels=channels,
            feature_maps=feature_maps,
            visualize=False,
            VISUAL=VISUAL
        )
