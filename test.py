'''
Test Model
'''
import os
import sys
import yaml
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K

pth_config = './config'
with open(os.path.join(pth_config, 'clef2016.yml'), 'r') as config_fl:
    config = yaml.load(config_fl)
pth_data = config['pth_data']
pth_utils = config['pth_utils']
pth_models = config['pth_models']
pth_weights = config['pth_weights']
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
from custom_common_resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet

from load_data import image_data_generator
from preprocess import center_crop
from util_visualize import check_loaded_data

if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Load and compile model. Model must be the same as the one used for training
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    classes = config['classes']
    backbone = config['backbone']
    att_type = config['att_type']


    if backbone == 'DenseNet':
        model = DenseNet169(
            include_top=False,
            weights='imagenet',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            pth_hist='',
            batch_size=2,
            att_type=att_type
        )
        preprocess_input = preprocess_input_densenet
    elif backbone == 'ResNet':
        model = ResNet152(
            include_top=False,
            weights='imagenet',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            pth_hist='',
            batch_size=2,
            att_type=att_type
        )
        preprocess_input = preprocess_input_resnet
    else:
        raise ValueError('Only ResNet and DenseNet backbone supported.')

    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])


    #Baseline DenseNet
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/91534'

    #BAM DenseNet
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/111326'

    #BAM DenseNet + reduction = 8
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/15257'

    #BAM DenseNet + reduction = 16, dilation_val = 2
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/15710'

    #BAM DenseNet + reduction = 4, dilation_val = 1
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/155833'

    #BAM DenseNet + reduction = 4, dilation_val = 2
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/16139'

    #BAM DenseNet + reduction = 8, dilation_val = 1
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/165823'

    #BAM DenseNet + reduction = 8, dilation_val = 2
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/16583'

    #BAM DenseNet + reduction = 16, dil =1
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/174827'

    #BAM DenseNet + reduction = 4, dil =4
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/174757'

    #Baseline ResNet
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/91513'

    #BAM ResNet dilation = 4, reduction = 16
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/111443'

    # BAM Resnet + dilation = 1, reduction = 16
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/192338'

    # BAM Resnet + dilation = 2, reduction = 16
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/19247'

    # BAM Resnet + dilation = 4, reduction = 8
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/202952'

    # BAM Resnet + dilation = 2, reduction = 8
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/203131'

    # BAM Resnet + dilation = 1, reduction = 8
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/214710'

    # BAM Resnet + dilation = 1, reduction = 4
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/214833'

    # #BAM ResNet + dilation 4, reduction 4
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/223924'

    #BAM ResNet + dilation 2, reduction 4
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/223853'


    # SE Bottleneck + No conv
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/123825'

    #SE Bottleneck + 3 x 3 conv
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/125124'

    # #SE Bottleneck + 1 x 1 conv
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/13192'

    #SE Bottleneck + red = 8
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/135018'

    #SE Bottleneck + red = 8 + conv = 1 x 1
    # weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-07-23/14519'

    #Retarget
    weight_name = '/media/jakep/Elements/ImageCLEF2016_weights/2020-08-08/133811'

    model.load_weights(os.path.join(os.path.join(weight_name,'cp-0030.ckpt')))

    #Load data for test
    ip_img_cols = config['ip_img_cols']
    ip_img_rows = config['ip_img_rows']
    batch_size = config['batch_size']

    TEST = os.path.join(pth_data, 'test')

    test_gen = image_data_generator(
        in_dir=TEST,
        preprocessing_function=preprocess_input,
        target_size=(ip_img_rows, ip_img_cols),
        batch_size=2,
        horizontal_flip=False,
        shuffle=False
    )
    steps = len(test_gen)
    test_gen = center_crop(test_gen, ip_img_rows, ip_img_cols, nw_img_cols, 2)
    predict = model.evaluate(test_gen, steps=steps, verbose=1)
    print(predict)
