'''
Train model
'''
import os
import sys
import random
from contextlib import redirect_stdout
import yaml
import numpy as np
#Load configuration file. Configuration file contains paths to other directories
pth_config = './config'
with open(os.path.join(pth_config, 'clef2016.yml'), 'r') as config_fl:
    config = yaml.load(config_fl)
pth_data = config['pth_data']
pth_utils = config['pth_utils']
pth_models = config['pth_models']
pth_weights = config['pth_weights']
pth_hist = config['pth_hist']
pth_visual = config['pth_visual']
pths_import = [
    pth_data,
    pth_utils,
    pth_models,
    pth_weights,
    pth_hist,
    pth_visual
]
for pth_import in pths_import:
    if pth_import not in sys.path:
        sys.path.append(pth_import)
#Import model for training
from custom_resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
#Load data from directory
from load_data import image_data_generator
#Import preprocessing functions
from preprocess import center_crop

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
from tensorflow.keras.callbacks import LearningRateScheduler

if __name__ == '__main__':
    #Set up tensorflow envirornment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print('Tensorflow addons version: {0}'.format(tfa.__version__))
    print("Num GPUs Used: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    #Set up directories
    weight_name = str(random.randint(0, 100000))
    WEIGHTS = os.path.join(pth_weights, weight_name)
    HISTORY = os.path.join(pth_hist, weight_name)
    VISUAL = os.path.join(pth_visual, weight_name)
    if not os.path.exists(WEIGHTS):
        os.mkdir(WEIGHTS)
    else:
        raise ValueError('Directory with the same name already exists!')
    #Save model weights
    print('Model weights for this run saved at: {0}'.format(WEIGHTS))
    #Save training history and model summary
    os.mkdir(HISTORY)
    print('Model Summary and training history for this run saved at: {0}'.format(HISTORY))
    #Save visualizations for this run
    os.mkdir(VISUAL)
    print('Visualizations for this run must be saved at: {0}'.format(VISUAL))

    #Load and compile model
    batch_size = config['batch_size']
    classes = config['classes']
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    model = ResNet152(
        use_bias=True,
        model_name='resnet152',
        weights='imagenet',
        input_shape=(nw_img_rows, nw_img_cols, 3),
        pooling='avg',
        classes=classes,
        pth_hist=HISTORY,
        batch_size=batch_size
    )
    model.compile(optimizer=SGDW(lr=1e-4, weight_decay=1e-6, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    print('Model compiled')
    with open(os.path.join(HISTORY, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print('Model input image resolution: ' + str(nw_img_cols) + 'x' + str(nw_img_cols))
    print('Model input batch_size: ' + str(batch_size))

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
        batch_size=batch_size,
        shuffle=True,
        horizontal_flip=False
    )
    steps_per_epoch = len(train_gen)
    #Center crop images
    train_gen = center_crop(
        generator=train_gen,
        height=ip_img_rows,
        width=ip_img_cols,
        crop_length=nw_img_cols,
        batch_size=batch_size
    )
    print('\n')

    #Load data for validation
    print('Validation Data:')
    VALID = os.path.join(pth_data, 'test')
    valid_gen = image_data_generator(
        in_dir=VALID,
        preprocessing_function=preprocess_input_resnet,
        target_size=(ip_img_rows, ip_img_cols),
        batch_size=2,
        horizontal_flip=False
    )
    steps = len(valid_gen)
    #Center crop images
    valid_gen = center_crop(
        generator=valid_gen,
        height=ip_img_rows,
        width=ip_img_cols,
        crop_length=nw_img_cols,
        batch_size=2
    )
    print('\n')

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

    #Train model
    history = model.fit(
        train_gen,
        shuffle=False,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        callbacks=[cp_callback]
    )

    #Save training history
    np.savez(
        os.path.join(HISTORY, 'history.npy'),
        history=history.history,
        epochs=history.epoch
    )
