'''
This module contains functions to either load all the data together before training
or load data from a image data generator
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

def load_data_categorical(
        in_dir='',
        num_classes=2,
        target_size=(224, 224)
    ):
    '''
    Description
    -----------
    Loads data for either training, validation or testing
    Entire data is loaded together in the RAM

    Args
    ----
    in_dir: path to train, validation or test directory
    num_classes: count of the number of classes to be classified
    target_size: resize original images to desired size

    Returns
    -------
    x_values: images to be used for either training, validation or testing
    y_values: one hot encoding of class
    '''
    x_values = []
    y_values = []
    #Sorting gives same order of y_values for either training, validation or testing
    classes = sorted([cl_name for cl_name in os.listdir(in_dir)])
    for label, cls in enumerate(classes):
        curr_dir = os.path.join(in_dir, cls)
        #Load each image in a class
        files = []
        for filename in os.listdir(curr_dir):
            if os.path.isfile(os.path.join(curr_dir, filename)):
                files.append(filename)
        files = sorted(files)
        for filename in files:
            path_file = os.path.join(curr_dir, filename)
            img = load_img(path_file, target_size=target_size, interpolation="lanczos")
            img = img_to_array(img)
            x_values.append(img)
            y_values.append(label)
    x_values = np.array(x_values).astype('float32')
    y_values = np.array(y_values).astype('float32')
    #Convert y_values to one hot format
    y_values = tf.keras.utils.to_categorical(y_values, num_classes)
    return x_values, y_values

def image_data_generator(
        in_dir='',
        preprocessing_function=None,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=32,
        shuffle=True,
        horizontal_flip=False
    ):
    '''
    Description
    -----------
    Loads data for either training, validation or testing

    Args
    ----
    Read: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    Returns
    -------
    generator: generator that contains batches of images and their labels
    '''
    img_gen = ImageDataGenerator(
        horizontal_flip=horizontal_flip,
        preprocessing_function=preprocessing_function
    )
    classes = sorted([cl_name for cl_name in os.listdir(in_dir)])
    generator = img_gen.flow_from_directory(
        in_dir,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
        interpolation='lanczos'
    )
    return generator
