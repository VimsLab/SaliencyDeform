'''
Utility functions to load data into model
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_categorical(in_dir='', num_classes=2, target_size=(224, 224)):
    '''
    Load images and labels
    '''
    x_values = []
    y_values = []
    #Labels are strings. sorted will work
    classes = sorted([cl_name for cl_name in os.listdir(in_dir)])
    for label, cls in enumerate(classes):
        curr_dir = os.path.join(in_dir, cls)
        files = sorted([filename for filename in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, filename))])
        for filename in files:
            path_file = os.path.join(curr_dir, filename)
            img = image.load_img(path_file, target_size=target_size, interpolation="lanczos")
            img = image.img_to_array(img)
            x_values.append(x_values)
            y_values.append(y_values)

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    y_values = tf.keray.utils.to_categorical(y_values, num_classes)
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
    Perform realtime augmentation of data
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
