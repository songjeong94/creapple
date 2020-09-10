# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Base libraries
import os
import random

# Keras libraries
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D, SeparableConv2D
from keras.layers import  Dropout, Activation, DepthwiseConv2D, GlobalAveragePooling2D, Reshape
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import plot_model
from keras.regularizers import l2

print(tf.__version__)

def relu6(x):
    return K.relu(x, max_value=6.0)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

_make_divisible(32 * 1.0, 8)

def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)

def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x

def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x

def MobileNetv2(k, input_shape = (256, 256, 3), alpha=1.0):
      inputs = Input(input_shape)

  first_filters = _make_divisible(32 * alpha, 8)
  x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

  x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
  x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
  x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
  x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
  x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
  x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
  x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

  if alpha > 1.0:
      last_filters = _make_divisible(1280 * alpha, 8)
  else:
      last_filters = 1280

  x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
  x = GlobalAveragePooling2D()(x)
  x = Reshape((1, 1, last_filters))(x)
  x = Dropout(0.3, name='Dropout')(x)
  x = Conv2D(k, (1, 1), padding='same')(x)

  x = Activation('softmax', name='softmax')(x)
  output = Reshape((k,))(x)

  model = Model(inputs, output)
  # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

  return model

mobile_net_v2_model = MobileNetv2(100, (256, 256, 3), 1.0)
mobile_net_v2_model.summary()

print(mobile_net_v2_model.layers[152].name)

print(mobile_net_v2_model.layers[114].name)
print(mobile_net_v2_model.layers[115].name)

print(mobile_net_v2_model.layers[53].name)
print(mobile_net_v2_model.layers[54].name)

print(mobile_net_v2_model.layers[27].name)
print(mobile_net_v2_model.layers[28].name)

print(mobile_net_v2_model.layers[11].name)

def mobile_unet(k, input_shape=(256, 256, 3)):
      input = Input(input_shape)

  mobile_net_v2 = MobileNetv2(k=k, input_shape=input_shape)
  mobile_net_v2_encoder_d1 = Model(
      inputs=mobile_net_v2.input,
      outputs=mobile_net_v2.get_layer(mobile_net_v2.layers[11].name).output)
  mobile_net_v2_encoder_d2 = Model(
      inputs=mobile_net_v2.input,
      outputs=mobile_net_v2.get_layer(mobile_net_v2.layers[28].name).output)
  mobile_net_v2_encoder_d3 = Model(
      inputs=mobile_net_v2.input,
      outputs=mobile_net_v2.get_layer(mobile_net_v2.layers[54].name).output)
  mobile_net_v2_encoder_d4 = Model(
      inputs=mobile_net_v2.input,
      outputs=mobile_net_v2.get_layer(mobile_net_v2.layers[115].name).output)
  mobile_net_v2_encoder_d5 = Model(
      inputs=mobile_net_v2.input,
      outputs=mobile_net_v2.get_layer(mobile_net_v2.layers[152].name).output)
  
  skip_4 = mobile_net_v2_encoder_d4(input)
  skip_3 = mobile_net_v2_encoder_d3(input)
  skip_2 = mobile_net_v2_encoder_d2(input)
  skip_1 = mobile_net_v2_encoder_d1(input)

  x = mobile_net_v2_encoder_d5(input)
  
  x = Conv2DTranspose(96, 4, strides=(2, 2), padding='same')(x)
  x = Add()([x, skip_4])
  x = _inverted_residual_block(x, 96, (3, 3), t=1, alpha=1.0, strides=1, n=1)
  
  x = Conv2DTranspose(32, 4, strides=(2, 2), padding='same')(x)
  x = Add()([x, skip_3])
  x = _inverted_residual_block(x, 32, (3, 3), t=1, alpha=1.0, strides=1, n=1)

  x = Conv2DTranspose(24, 4, strides=(2, 2), padding='same')(x)
  x = Add()([x, skip_2])
  x = _inverted_residual_block(x, 24, (3, 3), t=1, alpha=1.0, strides=1, n=1)

  x = Conv2DTranspose(16, 4, strides=(2, 2), padding='same')(x)
  x = Add()([x, skip_1])
  x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=1.0, strides=1, n=1)

  x = Conv2DTranspose(k, 4, strides=(2, 2), padding='same')(x)

  x = Activation("softmax")(x)

  model = Model(input, [x])
  return model

mu = mobile_unet(10)
mu.summary()

from keras.engine import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.regularizers import l2