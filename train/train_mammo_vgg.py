from __future__ import print_function
import os

import numpy as np
from keras.utils import np_utils
from models.mammo_vgg import VGG16
import utils.simple_loader as sl


if __name__ == '__main__':
    model = VGG16(include_top=True, weights=None)

    # Must use the RANDOM_SEED environment variable as specified in challenge guidelines.
    seed = os.getenv('RANDOM_SEED', '1337')
    np.random.seed(int(seed))

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

    sl = sl.SimpleLoader()

    labels = sl.labels
    labels = np_utils.to_categorical(labels, nb_classes)

    model.fit(sl.imgs, labels, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_split=train_split)

    # score = model.evaluate(test['x'], test['y'], verbose=0)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
