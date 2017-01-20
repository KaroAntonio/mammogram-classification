from __future__ import print_function

import os

import numpy as np
import utils.simple_loader as sl

import utils.constants as c
from models.mammo_vgg import VGG16

if __name__ == '__main__':
    model = VGG16(include_top=True, weights=None)

    # Must use the RANDOM_SEED environment variable as specified in challenge guidelines.
    seed = os.getenv('RANDOM_SEED', '1337')
    np.random.seed(int(seed))

    batch_size = 32
    nb_epoch = 5

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    # actual training set contains 500k+ images which would take
    # too long if we were to use the whole set.
    training_set_ratio = 1.0 if c.LOCAL_TEST else 0.05

    creator = sl.PNGBatchGeneratorCreator(c.PREPROCESS_IMG_DIR + '/', batch_size=batch_size)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'precision', 'recall', 'fmeasure'])
    model.summary()

    # Number of samples per epoch has to be a multiple of batch size. Thus we'll use the largest
    # multiple of batch size possible. This wastes at most batch size amount of samples.
    num_training_samples = (creator.total_training_samples() * training_set_ratio) // batch_size * batch_size
    history = model.fit_generator(creator.get_generator('training'), num_training_samples,
                                  nb_epoch, validation_data=creator.get_generator('validation'),
                                  nb_val_samples=creator.total_validation_samples())

    # Save metrics from the training process for later visualization
    metrics = ['binary_accuracy', 'recall', 'precision', 'fmeasure']
    with open(c.MODELSTATE_DIR + '/plot.txt', 'w') as f:
        for metric in metrics:
            f.write('train {}\n'.format(metric))
            f.write(', '.join(str(x) for x in history.history[metric]))
            f.write('\n')
            f.write('validation {}\n'.format(metric))
            f.write(', '.join(str(x) for x in history.history['val_' + metric]))
            f.write('\n\n\n')

    model.save(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)
    model.save_weights(c.MODELSTATE_DIR + '/' + c.WEIGHTS_FILENAME)
