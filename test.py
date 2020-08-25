'''
Test Model
'''
import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

pth_config = './config'
with open(os.path.join(pth_config, 'clef.yml'), 'r') as config_fl:
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

from custom_common_densenet import DenseNet121, DenseNet169
from custom_common_resnet import ResNet50, ResNet101
from inception import InceptionV3
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception

from load_data import image_data_generator
from preprocess import center_crop
from util_visualize import check_loaded_data
import sklearn.metrics as metrics

if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['run_on_gpu'])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print('Tensorflow version: {0}'.format(tf.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Load and compile model. Model must be the same as the one used for training
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    classes = config['classes']
    backbone = config['backbone']
    att_type = config['att_type']


    if backbone == 'DenseNet':
        model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            pth_hist='',
            batch_size=12,
            att_type=att_type
        )
        preprocess_input = preprocess_input_densenet
    elif backbone == 'ResNet':
        model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            pth_hist='',
            batch_size=12,
            att_type=att_type
        )
        preprocess_input = preprocess_input_resnet
    elif backbone == 'Inception':
        model = InceptionV3(include_top=False,
                            weights=None,
                            input_shape=(nw_img_rows, nw_img_cols, 3),
                            classes=classes,
                            batch_size=12,
                            pooling='avg',
                            att_type=att_type)
        preprocess_input = preprocess_input_inception
    else:
        raise ValueError('Only ResNet and DenseNet backbone supported.')

    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])


    # #2013 + DenseNet + 4
    # weight_name = '/media/jakep/Bio-Imaging VR/ImageCLEF2013_weights/2020-08-23/134927'#134949
    #
    #
    # #2013 + DenseNet + 4
    # weight_name = '/media/jakep/Bio-Imaging VR/ImageCLEF2013_weights/2020-08-23/18320'#183120'

    weight_name = '/media/jakep/Bio-Imaging VR/ImageCLEF2013_weights/2020-08-25/122258'

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
        batch_size=12,
        horizontal_flip=False,
        shuffle=False
    )
    steps = len(test_gen)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    test_gen = center_crop(test_gen, ip_img_rows, ip_img_cols, nw_img_cols, 2)
    # predict = model.evaluate(test_gen, steps=steps, verbose=1)

    predict = model.predict(test_gen, steps=steps, verbose=1)

    predicted_classes = np.argmax(predict, axis=1)
    report = metrics.classification_report(true_classes, predicted_classes, digits=5)# target_names=class_labels)
    print(report)
