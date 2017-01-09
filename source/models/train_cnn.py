# WARNING: This file is not currently used in any Docker images and likely won't work.

from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np

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

dl = DataLoader(m_fid, cw_fid, img_dir, num_imgs=20,scale=(img_dim,img_dim))

train,test = dl.get_train_test(split=train_split)

train['y'] = np_utils.to_categorical(train['y'],nb_classes)
test['y'] = np_utils.to_categorical(test['y'],nb_classes)

#  Format input for theano (th) or tensorflow (else)
if K.image_dim_ordering()=='th':
	train['x'] = np.expand_dims(train['x'],1)
	test['x'] = np.expand_dims(test['x'],1)
	input_shape = (1,img_dim, img_dim)
else:
	train['x'] = np.expand_dims(train['x'],3)
	test['x'] = np.expand_dims(test['x'],3)
	input_shape = (img_dim, img_dim,1)


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(300))  # this should be scaled up as the input_shape increases
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train['x'], train['y'], batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test['x'], test['y']))

score = model.evaluate(test['x'], test['y'], verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

