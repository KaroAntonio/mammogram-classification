from __future__ import print_function

import os

import numpy as np
import utils.simple_loader as sl

import utils.constants as c
from models.mammo_vgg import VGG16

from keras.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == '__main__':
    model = VGG16(include_top=True, weights=None)

    # Must use the RANDOM_SEED environment variable as specified in challenge guidelines.
    seed = os.getenv('RANDOM_SEED', '1337')
    np.random.seed(int(seed))

    batch_size = 32
    nb_epoch = 25
    validation_split = 0.25
    negative_ratio = 2.0

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    creator = sl.PNGBatchGeneratorCreator(c.PREPROCESS_IMG_DIR + '/',
                                          batch_size=batch_size, validation_split=validation_split)

    model_checkpoint = ModelCheckpoint(c.MODELSTATE_DIR + '/{epoch:02d}' + c.MODEL_FILENAME,
                                       monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                       mode='auto', period=1)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, verbose=1, mode='auto')

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'precision', 'recall', 'fmeasure'])
    model.summary()

    # Get a balanced dataset with a negative:positive ratio of approximately negative_ratio
    balanced = creator.balance_dataset(creator.get_dataset('training'), negative_ratio=negative_ratio)

    # Number of samples per epoch has to be a multiple of batch size. Thus we'll use the largest
    # multiple of batch size possible. This wastes at most batch size amount of samples.
    # Also limit training to 20000 images max due to time constraints.
    num_training_samples = min(20000, len(balanced.index)) // batch_size * batch_size
    num_validation_samples = num_training_samples * validation_split

    print('Training on set of {} images with a negative:positive ratio of approximately {}'.format(len(balanced.index),
                                                                                                   negative_ratio))
    history = model.fit_generator(creator.get_generator(dataset=balanced), num_training_samples,
                                  nb_epoch, validation_data=creator.get_generator('validation'),
                                  nb_val_samples=num_validation_samples,
                                  callbacks=[early_stopping, model_checkpoint])

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
