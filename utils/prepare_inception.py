'''
Train model
'''
import os
import sys
import yaml
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import save_model

#Load configuration file. Configuration file contains paths to other directories
pth_config = '../config'
with open(os.path.join(pth_config, 'clef.yml'), 'r') as config_fl:
    config = yaml.load(config_fl)
pth_models = config['pth_models']
if pth_models not in sys.path:
    sys.path.append(pth_models)
from inception import InceptionV3

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #Load and compile model
    batch_size = config['batch_size']
    classes = config['classes']
    nw_img_cols = config['nw_img_cols']
    nw_img_rows = config['nw_img_rows']

    model = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_shape=(nw_img_rows, nw_img_cols, 3),
                        batch_size=batch_size,
                        pooling=None,
                        by_name=False)
    print(model.summary())
    filepath ='/home/jakep/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop_by_name3.h5'
    save_model(model, filepath)
