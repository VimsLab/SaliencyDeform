'''
Utility functions to visualize feature_maps and input data to a model
'''
import os
import math
from itertools import cycle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
cycol = cycle('bgrcmk')

def visualize_feature_maps(
        num_img=16,
        layer_index=0,
        layer_name='input',
        channels=3,
        feature_maps=[],
        visualize=True
    ):
    '''
    Visualize feature map channels per batch.
    Few channels at the end would be discarded if square root of channels is not a whole number.
    '''
    if feature_maps == []:
        raise ValueError('Feature maps cannot be of NoneType.')
    if len(tf.shape(feature_maps)) != 4:
        raise ValueError('Feature maps must have 4 dimensions and must be in NHWC format.')
    grid_size = channels
    while math.sqrt(grid_size) != int(math.sqrt(grid_size)):
        grid_size += 1

    if not visualize:
        RESULTS = os.path.join('./results', str(layer_index))
        os.mkdir(RESULTS)
    else:
        matplotlib.use('TkAgg')

    for img in range(0, num_img, 1):
        fig = plt.figure()
        for channel in range(1, channels + 1, 1):
            axis = fig.add_subplot(math.sqrt(grid_size), math.sqrt(grid_size), channel)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.imshow(feature_maps[img, :, :, channel - 1], cmap='jet')
        if visualize:
            plt.show()
            plt.close()
        else:
            fig.savefig(os.path.join(RESULTS, str(img) + '_' + layer_name + '.png'))
            print('Saved')
            plt.close()


def check_loaded_data(
        imgs=[],
        lbls=[],
        grid_size=64
    ):
    '''
    Check labels for input images
    '''
    if imgs == [] or lbls == []:
        raise ValueError('Images or labels cannot be empty.')

    if len(imgs) != len(lbls):
        raise ValueError('Image and labels must be equal in length.')

    while math.sqrt(grid_size) != int(math.sqrt(grid_size)):
        grid_size += 1

    fig = plt.figure()
    for enum_index, img in enumerate(imgs):
        axis = fig.add_subplot(math.sqrt(grid_size), math.sqrt(grid_size), enum_index + 1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.imshow(img.astype('uint8'))
        axis.set_title(label=str(np.argmax(lbls[enum_index]) + 1), y=-0.01) #plus one is important
    plt.show()

def plot_history(
        histories={},
        max_epochs=0,
        key='accuracy'
    ):
    '''
    Utility function to plot model history paramerters
    '''
    if len(histories) <= 0:
        raise ValueError('Model name and history are required to plot graph')
    plt.figure(figsize=(10, 10))
    for history in histories:
        if history['history'].item()[key]:
            plt.plot(history['epoch'], history['history'].item()[key], color=next(cycol), label='{0} for {1}'.format(key, history['model_name']))
        if 'val_' + key in history['history'].item():
            plt.plot(history['epoch'], history['history'].item()['val_' + key], color=next(cycol), label='{0} for {1}'.format('val_' + key, history['model_name']))
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.xlim([0, max_epochs])
    plt.show()
