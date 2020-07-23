'''
This module contains functions to visualize feature_maps and input data to a model for sanity check
'''
import os
import math
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
cycol = cycle('bgrcmk')
import matplotlib.gridspec as gridspec

def visualize_feature_maps(
        num_img=16,
        layer_index=0,
        layer_name='input',
        channels=3,
        feature_maps=[],
        visualize=True,
        VISUAL=None
    ):
    '''
    Description
    -----------
    Visualize feature map for specified number channels for all images

    Args
    ----
    num_img: total number of images
    layer_index: index number of the layer
    layer_name: name of the layer
    channels: number of channels
    feature_maps: feature_maps of shape (B, H, W, C)
    visualize: If true feature maps are visualized
    VISUAL: If visualize is false, path where visualized results will be saved
    '''
    if feature_maps == []:
        raise ValueError('Feature maps cannot be of NoneType.')
    if len(tf.shape(feature_maps)) != 4:
        raise ValueError('Feature maps must have 4 dimensions and must be in NHWC format.')
    grid_size = channels
    while math.sqrt(grid_size) != int(math.sqrt(grid_size)):
        grid_size += 1

    if not visualize:
        VISUAL = os.path.join(VISUAL, str(layer_index))
        os.mkdir(VISUAL)
    else:
        #Faster
        matplotlib.use('TkAgg')
    print(VISUAL)
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
            fig.savefig(os.path.join(VISUAL, str(img) + '_' + layer_name + '.png'))
            plt.close()
            print('Saved')



def check_loaded_data(
        imgs=[],
        lbls=[],
        grid_size=64
    ):
    '''
    Description
    -----------
    Sanity check function to check images before training, validation or testing

    Args
    ----
    imgs: images of shape (B, H, W, C)
    lbls: corresponding label of an image in one hot format
    grid_size: sample size from the total number of the images in the entire data set
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
        axis.set_title(label=str(np.argmax(lbls[enum_index]) + 1)) #plus one is important
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
