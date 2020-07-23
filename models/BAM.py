import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

class BAM_ChannelGate(Layer):
    def __init__(self, reduction_ratio=16):
        super(BAM_ChannelGate, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, shape):
        channel_axis = -1
        channel = shape[channel_axis]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Reshape((1, 1, channel))
        self.gate_c_fc = layers.Dense(channel//self.reduction_ratio,
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.gate_c_bn = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)
        self.gate_c_relu = layers.Activation('relu')
        self.gate_c_fc_final = layers.Dense(channel,
                                            kernel_initializer='he_normal',
                                            use_bias=True,
                                            bias_initializer='zeros')

    def call(self, feature_maps):
        channel_axis = -1
        channel = feature_maps.shape[channel_axis]
        avg_pool = self.avg_pool(feature_maps)
        flatten = self.flatten(avg_pool)
        assert flatten.shape[1:] == (1, 1, channel)
        gate_c_fc = self.gate_c_fc(flatten)
        gate_c_bn = self.gate_c_bn(gate_c_fc)
        gate_c_relu = self.gate_c_relu(gate_c_bn)
        gate_c_fc_final = self.gate_c_fc_final(gate_c_relu)
        return gate_c_fc_final

class BAM_SpatialGate(Layer):
    def __init__(self, reduction_ratio=16, dilation_val=4):
        super(BAM_SpatialGate, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.dilation_val = dilation_val

    def build(self, shape):
        channel_axis = -1
        channel = shape[channel_axis]
        self.gate_s_conv_reduce = layers.Conv2D(filters=channel//self.reduction_ratio,
                                                kernel_size=(1, 1),
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                strides=(1, 1))
        self.gate_s_bn_reduce = layers.BatchNormalization(axis=channel_axis,
                                                          epsilon=1.001e-5)
        self.gate_s_relu_reduce = layers.Activation('relu')
        self.gate_s_conv_di_0 = layers.Conv2D(filters=channel//self.reduction_ratio,
                                              kernel_size=(3, 3),
                                              use_bias=True,
                                              kernel_initializer='glorot_uniform',
                                              strides=(1, 1),
                                              padding='same',
                                              dilation_rate=(self.dilation_val,
                                                             self.dilation_val))
        self.gate_s_bn_di_0 = layers.BatchNormalization(axis=channel_axis,
                                                        epsilon=1.001e-5)
        self.gate_s_relu_di_0 = layers.Activation('relu')

        self.gate_s_conv_di_1 = (layers.Conv2D(filters=channel//self.reduction_ratio,
                                               kernel_size=(3, 3),
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform',
                                               strides=(1, 1),
                                               padding='same',
                                               dilation_rate=(self.dilation_val,
                                                              self.dilation_val)))
        self.gate_s_bn_di_1 = (layers.BatchNormalization(axis=channel_axis,
                                                         epsilon=1.001e-5))
        self.gate_s_relu_di_1 = layers.Activation('relu')

        self.gate_s_conv_final = layers.Conv2D(filters=1,
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform',
                                               kernel_size=(1, 1),
                                               strides=(1, 1))

    def call(self, feature_maps):
        gate_s_conv_reduce = self.gate_s_conv_reduce(feature_maps)
        gate_s_bn_reduce = self.gate_s_bn_reduce(gate_s_conv_reduce)
        gate_s_relu_reduce = self.gate_s_relu_reduce(gate_s_bn_reduce)
        gate_s_conv_di_0 = self.gate_s_conv_di_0(gate_s_relu_reduce)
        gate_s_bn_di_0 = self.gate_s_bn_di_0(gate_s_conv_di_0)
        gate_s_relu_di_0 = self.gate_s_relu_di_1(gate_s_bn_di_0)
        gate_s_conv_di_1 = self.gate_s_conv_di_1(gate_s_relu_di_0)
        gate_s_bn_di_1 = self.gate_s_bn_di_1(gate_s_conv_di_1)
        gate_s_relu_di_1 = self.gate_s_relu_di_1(gate_s_bn_di_1)
        gate_s_conv_final = self.gate_s_conv_final(gate_s_relu_di_1)
        return gate_s_conv_final

class BAMLayer(Layer):
    def __init__(self, reduction_ratio=16, dilation_val=4):
        super(BAMLayer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.dilation_val = dilation_val

    def build(self, shape):
        self.channel_att = BAM_ChannelGate(self.reduction_ratio)
        self.spatial_att = BAM_SpatialGate(self.reduction_ratio, self.dilation_val)

    def call(self, feature_maps):
        channel_att = self.channel_att(feature_maps)
        spatial_att = self.spatial_att(feature_maps)
        att = layers.Add()([channel_att, spatial_att])
        att = layers.Activation('sigmoid')(att)
        att = tf.math.add(1.0, att)
        return att

def channel_gate(feature_maps, reduction_ratio=16):
    channel_axis = -1
    channel = feature_maps.shape[channel_axis]
    avg_pool = layers.GlobalAveragePooling2D()(feature_maps)
    flatten = layers.Reshape((1, 1, channel))(avg_pool)
    gate_c_fc = layers.Dense(channel//reduction_ratio,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                            )(flatten)
    gate_c_bn = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(gate_c_fc)
    gate_c_relu = layers.Activation('relu')(gate_c_bn)
    gate_c_fc_final = layers.Dense(channel,
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros',
                                  )(gate_c_relu)
    return gate_c_fc_final


def spatial_gate(feature_maps, reduction_ratio=16, dilation_val = 4):
    channel_axis = -1
    channel = feature_maps.shape[channel_axis]
    gate_s_conv_reduce = layers.Conv2D(filters=channel//reduction_ratio,
                                       kernel_size=(1, 1),
                                       use_bias=True,
                                       kernel_initializer='glorot_uniform',
                                       strides=(1, 1))(feature_maps)
    gate_s_bn_reduce = layers.BatchNormalization(axis=channel_axis,
                                                 epsilon=1.001e-5)(gate_s_conv_reduce)
    gate_s_relu_reduce = layers.Activation('relu')(gate_s_bn_reduce)
    gate_s_conv_di_0 = layers.Conv2D(filters=channel//reduction_ratio,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     dilation_rate=(dilation_val, dilation_val)
                                    )(gate_s_relu_reduce)
    gate_s_bn_di_0 = layers.BatchNormalization(axis=channel_axis,
                                               epsilon=1.001e-5)(gate_s_conv_di_0)
    gate_s_relu_di_0 = layers.Activation('relu')(gate_s_bn_di_0)
    gate_s_conv_di_1 = layers.Conv2D(filters=channel//reduction_ratio,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     padding='same',
                                     dilation_rate=(dilation_val, dilation_val)
                                    )(gate_s_relu_di_0)
    gate_s_bn_di_1 = layers.BatchNormalization(axis=channel_axis,
                                               epsilon=1.001e-5)(gate_s_conv_di_1)
    gate_s_relu_di_1 = layers.Activation('relu')(gate_s_bn_di_1)
    gate_s_conv_final = layers.Conv2D(filters=1,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      kernel_size=(1, 1),
                                      strides=(1, 1))(gate_s_relu_di_1)
    return gate_s_conv_final

def BAMBlock(feature_maps):
    channel_att = channel_gate(feature_maps)
    spatial_att = spatial_gate(feature_maps)
    att = layers.Add()([channel_att, spatial_att])
    att = layers.Activation('sigmoid')(att)
    att = tf.math.add(1.0, att)
    return att
