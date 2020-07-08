'''
Test Model
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
pths_import = [
    pth_data,
    pth_utils,
    pth_models,
    pth_weights,
]
for pth_import in pths_import:
    if pth_import not in sys.path:
        sys.path.append(pth_import)


from custom_common_resnet import ResNet50, ResNet101, ResNet152
# from common_resnet import ResNet50, ResNet101,ResNet152
# from common_densenet import DenseNet169
# from custom_common_densenet import DenseNet169

from load_data import image_data_generator
from preprocess import center_crop
from util_visualize import check_loaded_data

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K

if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))



    #Load and compile model. Model must be the same as the one used for training
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    classes = config['classes']
    weight_name = ''
    model = ResNet152(
        include_top=False,
        weights='imagenet',
        input_shape=(nw_img_rows, nw_img_cols, 3),
        pooling='avg',
        classes=classes,
        batch_size=2
    )
    model.compile(optimizer=SGDW(lr=0.0001, weight_decay=1e-6, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.load_weights(os.path.join('/media/jakep/Elements/3D Chemical Structures/3648','cp-0020.ckpt')).expect_partial()
    # model.load_weights(os.path.join('/media/jakep/Elements/3D Chemical Structures/23397','cp-0020.ckpt')).expect_partial()



    #Load data for test
    ip_img_cols = config['ip_img_cols']
    ip_img_rows = config['ip_img_rows']
    batch_size = config['batch_size']

    TEST = os.path.join(pth_data, 'Test')


    test_gen = image_data_generator(
        in_dir=TEST,
        preprocessing_function=preprocess_input_resnet,
        target_size=(ip_img_rows, ip_img_cols),
        batch_size=2,
        horizontal_flip=False,
        shuffle=False
    )
    steps = len(test_gen)
    test_gen = center_crop(test_gen, ip_img_rows, ip_img_cols, nw_img_cols, 2)
    predict = model.evaluate(test_gen, steps=steps, verbose=1)
    print(predict)
    # count = 0
    # img_proc = 0
    # while True:
    #     if img_proc == 247:
    #         break
    #     img_proc += 1
    #     print(img_proc)
    #     batch_x, batch_y = next(test_gen)
    #     # check_loaded_data(batch_x, batch_y, len(batch_x))
    #     start_y = (ip_img_rows - nw_img_rows) // 2
    #     start_x = (ip_img_cols - nw_img_cols) // 2
    #     batch_crops = batch_x[:, start_x:(ip_img_cols - start_x), start_y:(ip_img_rows - start_y), :]
    #     #If number of image per batch is divisible by batch size just yield them
    #
    #     predict = model.predict(batch_crops, verbose=1)
    #     res = K.equal(K.argmax(batch_y, axis=-1),
    #                                           K.argmax(predict, axis=-1))
    #
    #     if False in res:
    #         count += 1
    #         check_loaded_data(batch_x, batch_y, len(batch_x))
    #
    #         # import pdb; pdb.set_trace()
    # # loss, accuracy = predict
    # # print(loss, accuracy)
    # print(count)
