# -*- coding: utf-8 -*-
"""
VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Input, Activation
from keras.models import Model


def VGG16(include_top=True, weights='imagenet', input_tensor=None):
    """
    Instantiate the VGG16 architecture
    :param include_top: whether to include the 3 fully-connected
                        layers at the top of the network.
    :param weights:
    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                         to use as image input for the model.
    :return:             a Keras model instance.
    """

    # Determine proper input shape based on tensorflow or theano as a backend
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    # Block 1
    x = Convolution2D(16, 3, 3, border_mode='same', name='block1_conv1')(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Convolution2D(32, 3, 3, border_mode='same', name='block1_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #
    # # Block 2
    # x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv1')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv2')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #
    # # Block 3
    # x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv1')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv2')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv3')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #
    # # Block 4
    # x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv1')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv2')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv3')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    #
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        # x = Dense(16, name='fc1')(x)
        # x = PReLU()(x)
        # x = Dense(16, name='fc2')(x)
        # x = PReLU()(x)
        # x = Dense(32, activation='softmax')(x)
        # x = PReLU()(x)
        x = Dense(1, activation='sigmoid', name='predictions')(x)

    # Create model
    m = Model(img_input, x)

    # load weights
    if weights is not None:
        # TODO Load weights
        pass

    return m
