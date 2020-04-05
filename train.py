'''
Train model
'''

import os
import sys
import random
from contextlib import redirect_stdout
import yaml
import numpy as np
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
from util_visualize import check_loaded_data
from custom_resnet import ResNet152
from custom_callbacks import ValidationCallback, step_decay
from load_data import image_data_generator
from preprocess import center_crop

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet

if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Set up directories to save model weights and history
    weight_name = str(random.randint(0, 100000))
    WEIGHTS = os.path.join(pth_weights, weight_name)
    HISTORY = os.path.join(pth_hist, weight_name)
    if not os.path.exists(WEIGHTS):
        os.mkdir(WEIGHTS)
    else:
        raise ValueError('Directory with the same name already exists!')
    if not os.path.exists(HISTORY):
        os.mkdir(HISTORY)
    else:
        raise ValueError('Directory with the same name already exists!')
    print('Weights for this run saved at: {0}'.format(WEIGHTS))
    print('Model Summary and training history for this run saved at: {0}'.format(HISTORY))


    #Load model with model hyperparameters
    classes = config['classes']
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = ResNet152(
            use_bias=True,
            model_name='resnet152',
            weights='imagenet',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            pth_hist=HISTORY
            )
        model.compile(optimizer=SGDW(lr=1e-4, weight_decay=1e-5, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
    print('Model compiled')
    with open(os.path.join(HISTORY, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    #Load data for training
    print('Training Data:')
    batch_size = config['batch_size']
    ip_img_cols = config['ip_img_cols']
    ip_img_rows = config['ip_img_rows']
    TRAIN = os.path.join(pth_data, 'train')

    train_gen = image_data_generator(
        in_dir=TRAIN,
        preprocessing_function=preprocess_input_resnet,
        target_size=(ip_img_rows, ip_img_cols),
        batch_size=batch_size
    )
    steps_per_epoch = len(train_gen)
    train_gen = center_crop(train_gen, ip_img_rows, ip_img_cols, nw_img_cols)
    print('\n')

    #Load data for validation
    print('Validation Data:')
    VALID = os.path.join(pth_data, 'test')
    valid_gen = image_data_generator(
        in_dir=VALID,
        preprocessing_function=preprocess_input_resnet,
        target_size=(ip_img_rows, ip_img_cols),
        batch_size=batch_size
    )
    steps = len(valid_gen)
    valid_gen = center_crop(valid_gen, ip_img_rows, ip_img_cols, nw_img_cols)
    print('\n')
    print('Model input image resolution: ' + str(nw_img_cols) + 'x' + str(nw_img_cols))
    print('Model input batch_size: ' + str(batch_size))


    #Load Callbacks and training hyperparameters
    epochs = config['epochs']
    print('Training model for ' + str(epochs) + ' epoch/s')
    CHECKPOINT = os.path.join(WEIGHTS, 'cp-{epoch:04d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT,
        verbose=1,
        save_weights_only=True,
        period=1
    )
    model.save_weights(CHECKPOINT.format(epoch=0))
    # lrate_callback = LearningRateScheduler(step_decay)
    history = model.fit(
        train_gen,
        shuffle=True,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        callbacks=[cp_callback, ValidationCallback((valid_gen, steps))]
    )
    np.savez(
        os.path.join(HISTORY, 'history.npy'),
        history=history.history,
        epochs=history.epoch
    )
