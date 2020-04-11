from util_visualize import check_loaded_data
from random import randint
import numpy as np

def center_crop(
        batches=None,
        height=224,
        width=224,
        crop_length=224,
        batch_size=16
    ):
    '''
    center crop images in a ImageDataGenerator
    TODO:
    Add check if generator is parsed
    '''
    while True:
        batch_x, batch_y = next(batches)
        start_y = (height - crop_length) // 2
        start_x = (width - crop_length) // 2
        batch_crops = batch_x[:, start_x:(width - start_x), start_y:(height - start_y), :]
        if len(batch_x) % batch_size == 0:
            yield (batch_crops, batch_y)
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
            yield (batch_crops, batch_y)
