'''
This module contains custom code for quicker testing of models since the total number of image
are not always divisible by the batch size
'''
import os
import sys
import yaml
import numpy as np
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

from resnet import ResNet152
# from resnet import ResNet152

from load_data import load_data_categorical
from preprocess import center_crop

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K

if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    #conv5
    # run = '41694'
    # run = '85448'
    run = '79196'
    max_accuracy = 0
    #Load data for test
    classes = config['classes']
    ip_img_cols = config['ip_img_cols']
    ip_img_rows = config['ip_img_rows']
    TEST_NPY = os.path.join(pth_data, 'npy')

    if os.path.exists(os.path.join(TEST_NPY, 'test_img_' + str(ip_img_rows) + '.npy')):
        if os.path.exists(os.path.join(TEST_NPY, 'test_lbl_' + str(ip_img_rows) + '.npy')):
            test_img = np.load(os.path.join(TEST_NPY, 'test_img_' + str(ip_img_rows) + '.npy'))
            test_lbl = np.load(os.path.join(TEST_NPY, 'test_lbl_' + str(ip_img_rows) + '.npy'))
    else:
        TEST = os.path.join(pth_data, 'test')
        test_img, test_lbl = load_data_categorical(
            in_dir=TEST,
            num_classes=classes,
            target_size=(ip_img_rows, ip_img_cols)
        )
        np.save(os.path.join(TEST_NPY, 'test_img_' + str(ip_img_rows) + '.npy'), test_img)
        np.save(os.path.join(TEST_NPY, 'test_lbl_' + str(ip_img_rows) + '.npy'), test_lbl)
    #preprocess input for resnet
    test_img = preprocess_input_resnet(test_img)
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    #Center crop images
    start_y = (ip_img_rows - nw_img_rows) // 2
    start_x = (ip_img_cols - nw_img_cols) // 2
    test_img = test_img[:, start_x:(ip_img_cols - start_x), start_y:(ip_img_rows - start_y), :]

    batch_size = config['batch_size']
    batch_deficit = test_img.shape[0] % batch_size
    batch_size_test_img = test_img[0:test_img.shape[0]-batch_deficit, :, :, :]
    batch_deficit_test_img = test_img[test_img.shape[0]-batch_deficit: test_img.shape[0], :, :, :]

    for cp_index in range(90, 91):
        weight_name = os.path.join(pth_weights, '{0}/cp-00{1}.ckpt'.format(run, str(cp_index)))
        model = None
        model = ResNet152(
            use_bias=True,
            model_name='resnet152',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            batch_size=batch_size,
        )
        model.compile(optimizer=SGDW(lr=1e-4, weight_decay=1e-5, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        model.load_weights(weight_name).expect_partial()

        predict_batch_size = model.predict(batch_size_test_img, verbose=2)

        model = ResNet152(
            use_bias=True,
            model_name='resnet152',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            batch_size=batch_deficit,
        )
        model.compile(optimizer=SGDW(lr=1e-4, weight_decay=1e-5, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        model.load_weights(weight_name).expect_partial()

        predict_batch_deficit = model.predict(batch_deficit_test_img, verbose=2)
        predict = tf.concat([predict_batch_size, predict_batch_deficit], 0)

        categorical_accuracy = K.cast(K.equal(K.argmax(test_lbl, axis=-1),
                                              K.argmax(predict, axis=-1)),
                                      K.floatx())
        categorical_accuracy = tf.math.reduce_sum(
            categorical_accuracy
        )

        categorical_accuracy = categorical_accuracy/test_img.shape[0]
        print(weight_name, ' : ', categorical_accuracy)
