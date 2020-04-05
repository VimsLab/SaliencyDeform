import os
import sys
import yaml
import numpy as np
from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, Conv3D, BatchNormalization, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Add, Reshape, Activation, Lambda, GlobalAveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf
from batch_deform import BatchRetarget
from tensorflow.keras import Model, Sequential
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
import random
pth_config = '../config'
with open(os.path.join(pth_config, 'clef2016.yml') , 'r') as config_fl:
  config = yaml.load(config_fl)
pth_data = config['pth_data']
pth_utils = config['pth_utils']
pth_models = config['pth_models']
pth_weights = config['pth_weights']
pth_hist = config['pth_hist']
pths_import = [
    pth_data,
    pth_utils,
    pth_models,
    pth_weights,
    pth_hist
]
for pth_import in pths_import:
    if pth_import not in sys.path:
        sys.path.append(pth_import)
from visualize import plot_history, visualize_feature_maps
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
batch_size = config['batch_size']
epoch = config['epoch']
ip_img_cols = config['ip_img_cols']
ip_img_rows = config['ip_img_rows']
nw_img_cols = config['nw_img_cols']
nw_img_rows = config['nw_img_rows']
classes = config['classes']

TRAIN = os.path.join(pth_data, 'npy')
weight_name = str(random.randint(0, 100000))
WEIGHTS = os.path.join(pth_weights, weight_name)
HISTORY = os.path.join(pth_hist, weight_name)
if not os.path.exists(WEIGHTS):
    os.mkdir(WEIGHTS)
else:
    raise ValueError('Directory with the same name already exists!')
if not os.path.exists(HISTORY):
    os.mkdir(HISTORY)
else:
    raise ValueError('Directory with the same name already exists!')




print(TRAIN)
if os.path.exists(os.path.join(TRAIN, 'train_img_' + str(nw_img_rows) + '.npy')) and os.path.exists(os.path.join(TRAIN, 'train_lbl_' + str(nw_img_rows) + '.npy')):
    train_img = np.load(os.path.join(TRAIN, 'train_img_' + str(nw_img_rows) + '.npy'))
    train_lbl = np.load(os.path.join(TRAIN, 'train_lbl_' + str(nw_img_rows) + '.npy'))

train_img = train_img/255.

img_input = Input((nw_img_rows,nw_img_rows,3))
x_salient = Conv2D(10, 3, strides=3, use_bias=False)(img_input)
x_salient = Activation('relu')(x_salient)
x_salient = Conv2D(10, 3, strides=3, use_bias=False)(x_salient)
x_salient = Activation('relu')(x_salient)
x_salient = Conv2D(1, 3, 1, use_bias=False)(x_salient)
x = Conv2D(64, 3, strides=1, use_bias= True)(img_input)
x = BatchRetarget()([x, x_salient])
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, 3, strides=1, use_bias=True)(x)
x = Flatten(name='avg_pool')(x)
x = Dense(classes, activation='softmax')(x)
model = Model(inputs = img_input, outputs = x)
model.compile(optimizer=SGDW(lr=1e-4, weight_decay=1e-5, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

print(model.summary())
# import pdb; pdb.set_trace()
CHECKPOINT = os.path.join(WEIGHTS,'cp-{epoch:04d}.ckpt')
print(CHECKPOINT)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT,
    verbose=1,
    save_weights_only=True,
    period=1
)

model.save_weights(CHECKPOINT.format(epoch=0))
history = model.fit(train_img, train_lbl,  shuffle = True, batch_size = batch_size, epochs=epoch, verbose = 1,callbacks=[cp_callback])

# model.load_weights('/home/ddot/document/clef16/weights/23680/cp-0001.ckpt').expect_partial()
# model = Model(inputs = model.input, outputs = model.layers[0].output)
# print(model.summary())
#
# feature_maps = model.predict(train_img[0:32,:,:,:])
# print(feature_maps.shape)
# print('Predicted')
# visualize_feature_maps(batch_size = batch_size, channels = feature_maps.shape[3], feature_maps = feature_maps, visualize = True)
