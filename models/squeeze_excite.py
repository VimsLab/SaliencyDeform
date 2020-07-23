from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

class SELayer(Layer):
    def __init__(self, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.reduction_ratio = reduction_ratio


    def build(self, shape):
        channel_axis = -1
        channel = shape[channel_axis]

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Reshape((1, 1, channel))
        self.fc_channel = layers.Dense(channel // self.reduction_ratio,
                                       activation='relu',
                                       kernel_initializer='he_normal',
                                       use_bias=False)
        self.fc_final = layers.Dense(channel,
                                     activation='sigmoid',
                                     kernel_initializer='he_normal',
                                     use_bias=False)

    def call(self, feature_maps):
        channel_axis = -1
        channel = feature_maps.shape[channel_axis]

        avg_pool = self.avg_pool(feature_maps)
        flatten = self.flatten(avg_pool)
        assert flatten.shape[1:] == (1, 1, channel)
        fc_channel = self.fc_channel(flatten)
        assert fc_channel.shape[1:] == (1, 1, channel//self.reduction_ratio)
        fc_final = self.fc_final(fc_channel)
        assert fc_final.shape[1:] == (1, 1, channel)
        return fc_final


def SEBlock(feature_maps):
    channel_axis = -1
    reduction_ratio = 16
    channel = feature_maps.shape[channel_axis]

    se_feature = layers.GlobalAveragePooling2D()(feature_maps)
    se_feature = layers.Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    se_feature = layers.Dense(channel // reduction_ratio,
                              activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=False)(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel//reduction_ratio)
    se_feature = layers.Dense(channel,
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)

    return se_feature
