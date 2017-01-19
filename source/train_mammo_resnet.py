from __future__ import print_function

import os

import numpy as np
import utils.simple_loader as sl

import utils.constants as c
from models.mammo_resnet import ResnetBuilder

if __name__ == '__main__':

	model = ResnetBuilder.build_resnet_18((3, 224, 224), 1)

	# Must use the RANDOM_SEED environment variable as specified in challenge guidelines.
	seed = os.getenv('RANDOM_SEED', '1337')
	np.random.seed(int(seed))

	batch_size = 10
	nb_epoch = 10

	creator = sl.PNGBatchGeneratorCreator(c.PREPROCESS_IMG_DIR + '/', batch_size=batch_size)

	model.compile(optimizer='adam', loss='binary_crossentropy',
				  metrics=['binary_accuracy', 'precision', 'recall'])
	model.summary()

	# Number of samples per epoch has to be a multiple of batch size. Thus we'll use the largest
	# multiple of batch size possible. This wastes at most batch size amount of samples.
	num_training_samples = creator.total_training_samples() // batch_size * batch_size
	model.fit_generator(creator.get_generator('training'), num_training_samples,
						nb_epoch, validation_data=creator.get_generator('validation'),
						nb_val_samples=creator.total_validation_samples())

	model.save(c.MODELSTATE_DIR + '/' + c.MODEL_FILENAME)
	model.save_weights(c.MODELSTATE_DIR + '/' + c.WEIGHTS_FILENAME)
