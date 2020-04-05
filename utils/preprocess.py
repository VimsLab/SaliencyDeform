from util_visualize import check_loaded_data

def center_crop(
        batches=16,
        height=224,
        width=224,
        crop_length=224
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
        yield (batch_crops, batch_y)
