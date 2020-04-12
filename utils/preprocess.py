'''
This module contains functions to preprocess the input before training the network
'''
#Import visualization function to check data loaded from a generator
from util_visualize import check_loaded_data
from random import randint
import numpy as np
def center_crop(
        generator=None,
        height=224,
        width=224,
        crop_length=224,
        batch_size=16
    ):
    '''
    Description
    -----------
    Center crop images

    Args
    ----
    generator: generator that contains batches of images and their labels
    height: orginal height of all images
    width: orginal weight of all images
    crop_length: height and width to be cropped to
    batch_size: number of images to be used for either training, validation or testing for a step per epoch

    Returns
    -------
    batch_crops: center cropped images
    batch_y: label of center cropped images
    '''
    while True:
        batch_x, batch_y = next(generator)
        # check_loaded_data(batch_x, batch_y, len(batch_x))
        start_y = (height - crop_length) // 2
        start_x = (width - crop_length) // 2
        batch_crops = batch_x[:, start_x:(width - start_x), start_y:(height - start_y), :]
        #If number of image per batch is divisible by batch size just yield them
        if len(batch_x) % batch_size == 0:
            # check_loaded_data(batch_crops, batch_y, len(batch_x))
            yield (batch_crops, batch_y)

        #Add images to a batch so it can be divisible by the batch size
        #Alternative would have been to drop remainder but it will reduces the number
        #of training images per epoch whioch might lead to lower test accuracy
        else:
            #Makes sure that images not divisible by batch size do not get discarded
            deficit = batch_size - len(batch_x)
            while deficit != 0:
                resample_index = randint(0, len(batch_x) -1)
                batch_crops = np.concatenate(
                    [batch_crops, np.expand_dims(batch_crops[resample_index, :, :, :],
                                                 axis=0)],
                    axis=0
                )
                batch_y = np.concatenate(
                    [batch_y, np.expand_dims(batch_y[resample_index, :],
                                             axis=0)],
                    axis=0
                )
                deficit = deficit - 1
            # check_loaded_data(batch_crops, batch_y, len(batch_x))
            yield (batch_crops, batch_y)
