import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class CBAM_SpatialGate(Layer):
    def __init__(self, kernel_size=7):
        super(CBAM_SpatialGate, self).__init__()
        self.kernel_size = kernel_size

    def build(self, shape):
        channel_axis = -1
        channel = shape[channel_axis]
        self.avg_pool = layers.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))
        self.max_pool = layers.Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))
        self.spatial = layers.Conv2D(filters=1,
                                     kernel_size=self.kernel_size,
                                     strides=1,
                                     padding='same',
                                     activation='sigmoid',
                                     kernel_initializer='he_normal',
                                     use_bias=False)

    def call(self, feature_maps):
        channel_axis=-1
        avg_pool = self.avg_pool(feature_maps)
        assert avg_pool.shape[-1] == 1
        max_pool = self.max_pool(feature_maps)
        assert max_pool.shape[-1] == 1
        concat = layers.Concatenate(axis=channel_axis)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        spatial_att = self.spatial(concat)
        assert spatial_att.shape[-1] == 1
        return spatial_att

class CBAM_ChannelGate(Layer):

    def __init__(self, reduction_ratio=16):
        super(CBAM_ChannelGate, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, shape):
        channel_axis = -1
        channel = shape[channel_axis]
        self.shared_layer_one = layers.Dense(channel//self.reduction_ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_layer_two = layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Reshape((1, 1, channel))
        self.max_pool = layers.GlobalMaxPooling2D()
        self.activation = layers.Activation('sigmoid')

    def call(self, feature_maps):
        channel_axis = -1
        channel = feature_maps.shape[channel_axis]

        avg_pool = self.avg_pool(feature_maps)
        avg_pool = self.flatten(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)
        avg_pool = self.shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel//self.reduction_ratio)
        avg_pool = self.shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)

        max_pool = self.max_pool(feature_maps)
        max_pool = self.flatten(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)
        max_pool = self.shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel//self.reduction_ratio)
        max_pool = self.shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)
        add = layers.Add()([avg_pool, max_pool])
        channel_att = self.activation(add)
        return channel_att

class CBAMLayer(Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7):
        super(CBAMLayer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, shape):
        self.channel_att = CBAM_ChannelGate(self.reduction_ratio)
        self.spatial_att = CBAM_SpatialGate(self.kernel_size)

    def call(self, feature_maps):
        cbam_features = self.channel_att(feature_maps)
        cbam_features = layers.multiply([feature_maps, cbam_features])
        cbam_features = self.spatial_att(cbam_features)
        cbam_features = layers.multiply([feature_maps, cbam_features])
        return cbam_features
