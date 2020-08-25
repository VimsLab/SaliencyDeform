'''
Train model
'''
import os
import sys
import random
from contextlib import redirect_stdout
import yaml
import numpy as np
from datetime import datetime
from collections import Counter
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

#Load configuration file. Configuration file contains paths to other directories
pth_config = './config'
with open(os.path.join(pth_config, 'clef.yml'), 'r') as config_fl:
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
from custom_common_densenet import DenseNet121, DenseNet169, DenseNet201
from custom_common_resnet import ResNet50, ResNet101
from inception import InceptionV3
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
#Load data from directory
from load_data import image_data_generator
#Import preprocessing functions
from preprocess import center_crop

if __name__ == '__main__':
    #Set seed value for reproducibility of results
    seed_value = 1

    # Set random generators to fixed seed value
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    #Set up python, tensorflow and CUDA envirornment
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['run_on_gpu'])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    print('Tensorflow version: {0}'.format(tf.__version__))
    print("GPUs Available: {0}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    if len(tf.config.experimental.list_physical_devices('GPU')) < 1:
        raise ValueError('GPU not found!')

    #Set up directories
    date = str(datetime.now().date())
    hour = str(datetime.now().time().hour)
    minute = str(datetime.now().time().minute)
    sec = str(datetime.now().time().second)

    if not os.path.exists(os.path.join(pth_hist, date)):
        os.mkdir(os.path.join(pth_hist, date))
        os.mkdir(os.path.join(pth_weights, date))
        os.mkdir(os.path.join(pth_visual, date))

    HISTORY = os.path.join(os.path.join(pth_hist, date), hour+minute+sec)
    os.mkdir(HISTORY)
    print('Model Summary and training history for this run saved at: {0}'.format(HISTORY))

    WEIGHTS = os.path.join(os.path.join(pth_weights, date), hour+minute+sec)
    os.mkdir(WEIGHTS)
    print('Model weights for this run saved at: {0}'.format(WEIGHTS))

    VISUAL = os.path.join(os.path.join(pth_visual, date), hour+minute+sec)
    os.mkdir(VISUAL)
    print('Visualizations for this run must be saved at: {0}'.format(VISUAL))

    #Load and compile model
    batch_size = config['batch_size']
    classes = config['classes']
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']
    backbone = config['backbone']
    att_type = config['att_type']

    if backbone == 'DenseNet':
        model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(nw_img_rows, nw_img_cols, 3),
            pooling='avg',
            classes=classes,
            pth_hist=HISTORY,
            batch_size=batch_size,
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
            pth_hist=HISTORY,
            batch_size=batch_size,
            att_type=att_type
        )
        preprocess_input = preprocess_input_resnet
    elif backbone == 'Inception':
        model = InceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=(nw_img_rows, nw_img_cols, 3),
                            classes=classes,
                            batch_size=batch_size,
                            pooling='avg',
                            by_name=True,
                            att_type=att_type)
        preprocess_input = preprocess_input_inception
    else:
        raise ValueError('Only Inception, ResNet and DenseNet backbone supported.')

    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    print('Model compiled')

    with open(os.path.join(HISTORY, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    plot_model(model, to_file=os.path.join(HISTORY, 'model.png'), dpi=300)

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
        preprocessing_function=preprocess_input,
        target_size=(ip_img_rows, ip_img_cols),
        batch_size=batch_size,
        shuffle=True,
        seed_value=seed_value,
        horizontal_flip=False
    )
    counter = Counter(train_gen.classes)
    steps_per_epoch = len(train_gen)

    #Center crop images
    train_gen = center_crop(
        generator=train_gen,
        height=ip_img_rows,
        width=ip_img_cols,
        crop_length=nw_img_cols,
        batch_size=batch_size,
        discard_end=False
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

    max_val = float(max(counter.values()))
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

    # Train model
    history = model.fit(
        train_gen,
        shuffle=False,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        callbacks=[cp_callback],
        # class_weight=class_weights
    )

    #Save training history
    np.savez(
        os.path.join(HISTORY, 'history.npy'),
        history=history.history,
        epochs=history.epoch
    )
