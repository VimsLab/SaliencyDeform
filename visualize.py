'''
Visual output of feature_maps from a forward pass
'''
import os
import sys
import yaml
pth_config = './config'
with open(os.path.join(pth_config, 'clef.yml'), 'r') as config_fl:
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

from custom_common_densenet import DenseNet121

from load_data import image_data_generator
from preprocess import center_crop
from util_visualize import visualize_feature_maps

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.optimizers import Adam
if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Load and compile model. Model must be the same as the one used for training
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    classes = config['classes']
    batch_size = config['batch_size']
    att_type = config['att_type']

    pth_weights = '/media/jakep/Bio-Imaging VR/ImageCLEF2013_weights/2020-08-23/18320'
    pth_visual = '/home/jakep/document/clef/visual/2020-08-23/18320'
    weight_name = os.path.join(pth_weights,'cp-0030.ckpt')
    model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(nw_img_rows, nw_img_cols, 3),
        pooling='avg',
        classes=classes,
        pth_hist='',
        batch_size=4,
        att_type=att_type
    )


    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.load_weights(weight_name)
    #name of layers to be visualized
    layer_names = {
        'pool2_pool',
        'normalize',
        'retarget',
        'pool3_pool',
        'normalize_1',
        'retarget_1',
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
            batch_size=4,
            horizontal_flip=False,
            shuffle=False
        )
        steps = len(visual_gen)
        visual_gen = center_crop(visual_gen, ip_img_rows, ip_img_cols, nw_img_cols, 4)
        layer_name = model.layers[visualize_layer_at].name
        model_partial = Model(inputs=model.input, outputs=model.layers[visualize_layer_at].output)
        print('Partial model compiled with its last layer {0} compiled'.format(layer_name))
        feature_maps = model_partial.predict(visual_gen, steps=steps)
        print('Predicted')
        channels = 4
        # import numpy as np
        feature_maps = feature_maps[0:20,:,:,:]
        num_img = feature_maps.shape[0]
        VISUAL = pth_visual
        visualize_feature_maps(
            num_img=num_img,
            layer_index=visualize_layer_at,
            layer_name=layer_name,
            channels=channels,
            feature_maps=feature_maps,
            visualize=False,
            VISUAL=VISUAL
        )
