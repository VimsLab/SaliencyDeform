'''
Visual Model Layers
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
pth_hist = config['pth_hist']
pths_import = [
    pth_data,
    pth_utils,
    pth_models,
    pth_weights,
    pth_hist
]
for pth_import in pths_import:
    if pth_import not in sys.path:
        sys.path.append(pth_import)

from custom_resnet import ResNet152
from load_data import image_data_generator
from preprocess import center_crop
from util_visualize import visualize_feature_maps

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
from tensorflow.keras import Model, Sequential


if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Load model with model hyperparameters
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    classes = config['classes']
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = ResNet152(
            use_bias=True,
            model_name='resnet152',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes)
        model.compile(optimizer=SGDW(lr=1e-5, weight_decay=1e-5, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        model.load_weights('/home/ddot/document/clef16/weights/38452/cp-0110.ckpt').expect_partial()

    visualize_layers_at = [0,1,2, 3, 4, 5, 6,7,8,9]
    for visualize_layer_at in visualize_layers_at:
        #Load data for visualization
        ip_img_cols = config['ip_img_cols']
        ip_img_rows = config['ip_img_rows']
        batch_size = config['batch_size']
        VISUALIZE = os.path.join(pth_data, 'viz')
        visual_gen = image_data_generator(
            in_dir=VISUALIZE,
            preprocessing_function=preprocess_input_resnet,
            target_size=(ip_img_rows, ip_img_cols),
            batch_size=batch_size,
            horizontal_flip=False,
            shuffle=False
        )
        steps = len(visual_gen)
        visual_gen = center_crop(visual_gen, ip_img_rows, ip_img_cols, nw_img_cols)

        layer_name = model.layers[visualize_layer_at].name
        model_partial = Model(inputs=model.input, outputs=model.layers[visualize_layer_at].output)
        # print(model_partial.summary())
        feature_maps = model_partial.predict(visual_gen, steps=steps)
        print(feature_maps.shape)
        print('Predicted')
        if feature_maps.shape[3] > 64:
            channels = feature_maps.shape[3]//64
        elif feature_maps.shape[3] == 64:
            channels = feature_maps.shape[3]//16
        else:
            channels = feature_maps.shape[3]
        visualize_feature_maps(
            num_img=20,
            layer_index=visualize_layer_at,
            layer_name=layer_name,
            channels=channels,
            feature_maps=feature_maps,
            visualize=False
        )
