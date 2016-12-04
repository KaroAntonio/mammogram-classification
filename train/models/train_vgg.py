# -*- coding: utf-8 -*-
'''VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''
from __future__ import print_function

from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input
from keras.models import Model

from train.utils.loaders import *

def VGG16(include_top=True, weights='imagenet',
          input_tensor=None):
    '''Instantiate the VGG16 architecture
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

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
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights is not None:
        # Load weights
        pass

    return model


if __name__ == '__main__':
    model = VGG16(include_top=True, weights=None)

    np.random.seed(1337)

    m_fid = 'data/exams_metadata_pilot.tsv'
    cw_fid = 'data/images_crosswalk_pilot.tsv'
    img_dir = 'data/pilot_images/'
    img_dim = 28 # height and width of images to scale to (og: ~3000x3000)
    train_split = 0.6
    batch_size = 5
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    nb_classes = 2

    nb_epoch = 12

    dl = DataLoader(m_fid, cw_fid, img_dir, num_imgs=20, scale=(img_dim,img_dim))

    train,test = dl.get_train_test(split=train_split)

    train['y'] = np_utils.to_categorical(train['y'],nb_classes)
    test['y'] = np_utils.to_categorical(test['y'],nb_classes)

    model.fit(train['x'], train['y'], batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test['x'], test['y']))

    score = model.evaluate(test['x'], test['y'], verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

