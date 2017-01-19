from __future__ import print_function

import csv
import os
import subprocess
import StringIO

import numpy as np
import pandas as pd
import keras.preprocessing.image as kimage
from PIL import Image

import constants as c


def load_delim_txt(fid, delim, fieldnames):
    f = open(fid, 'r')
    reader = csv.DictReader(f, delimiter=delim, fieldnames=fieldnames)
    d = []
    for row in reader:
        d += [row]
    f.close()
    return d


class SimpleLoader:
    """
    Used for loading preprocessed PNG files.
    """

    def __init__(self):
        labelled = load_delim_txt(c.PREPROCESS_DIR + '/metadata/image_labels.txt',
                                  ' ', ('img', 'label'))

        self.imgs = []
        self.labels = []
        for pair in labelled:
            loaded = kimage.load_img(c.PREPROCESS_IMG_DIR + '/' + pair['img'], target_size=(224, 224))
            # Loads the image with the correct dim_ordering for the backend by default.
            loaded = kimage.img_to_array(loaded)
            self.imgs.append(loaded)
            self.labels.append(pair['label'])

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)


class BatchGeneratorCreator(object):
    """
    Abstract base class used for batch loading image files.
    Batch loading is required because there could be hundreds of thousands of images.
    """

    def __init__(self, imgs_dir, validation_split=0.25, batch_size=500):
        # imgs_dir should be the path to directory containing the image files,
        # including a trailing slash.
        self.imgs_dir = imgs_dir
        # Proportion of samples to use for validation.
        self.validation_split = validation_split
        self.batch_size = batch_size

        fields = ['filename', 'cancer']
        self.img_metadata = pd.read_csv(c.IMAGES_CROSSWALK_FILEPATH, sep="\t",
                                        na_values='.', usecols=fields)
        self.training_metadata = self.img_metadata[:self.total_training_samples()]
        self.validation_metadata = self.img_metadata[self.total_training_samples():]

    def total_samples(self):
        return len(self.img_metadata.index)

    def total_training_samples(self):
        return int(self.total_samples() * (1 - self.validation_split))

    def total_validation_samples(self):
        return self.total_samples() - self.total_training_samples()

    def _get_dataset(self, dataset):
        if dataset == 'all':
            return self.img_metadata
        elif dataset == 'training':
            return self.training_metadata
        else:
            return self.validation_metadata

    def get_generator(self, dataset='all', train_mode=True):
        metadata_frame = self._get_dataset(dataset)

        curr_idx = 0
        dataset_len = len(metadata_frame.index)

        # A batch cannot be larger than the total number of samples in the dataset.
        if self.batch_size > dataset_len:
            raise ValueError('Batch size {} is larger than the number of samples {} in the dataset.'
                             .format(self.batch_size, dataset_len))

        # Keras expects the generator to output samples infinitely.
        # Ideally we'll choose a number of samples to train on so we don't wrap around but
        # we still need to handle that case.
        while 1:
            num_added = 0
            x = np.empty((self.batch_size, c.WIDTH, c.HEIGHT, 3))
            if train_mode:
                y = np.empty(self.batch_size)

            # Need to ensure that we wrap around when we reach the end.
            wraparound_idx = -1
            end_idx = curr_idx + self.batch_size
            if end_idx > dataset_len:
                wraparound_idx = end_idx - dataset_len
                end_idx = dataset_len

            # Read in dcm files and load the converted pixel values into the x array.
            # Load the cancer labels into the y array.
            iter_range = range(curr_idx, end_idx) + range(0, wraparound_idx)
            for i in iter_range:
                row = metadata_frame.iloc[i]
                img_file = self._get_image_data(self.imgs_dir + row['filename'])
                x[num_added] = kimage.img_to_array(self._process_img_data(img_file))

                if train_mode:
                    y[num_added] = int(row['cancer'])
                curr_idx += 1
                num_added += 1

                if curr_idx >= dataset_len:
                    curr_idx = 0

            if train_mode:
                yield x, y
            else:
                yield x

    def _get_image_data(self, filename):
        # Must override this in base class. filename is a string containing the path to an image
        # file. Return a Pillow Image object.
        raise NotImplementedError

    def _process_img_data(self, img):
        # Must override this in base class. img is a Pillow Image object. Must return a
        # Pillow Image object.
        return img


class PNGBatchGeneratorCreator(BatchGeneratorCreator):
    """
    Used for batch loading 8-bit RGB PNG files.
    """

    def _get_image_data(self, filename):
        real_name = filename.replace('.dcm', '.png')
        return Image.open(real_name)


class DICOMBatchGeneratorCreator(BatchGeneratorCreator):
    """
    Used for batch loading raw DICOM files.
    """
    def __init__(self, imgs_dir, validation_split=0.25, batch_size=500):
        super(DICOMBatchGeneratorCreator, self).__init__(imgs_dir, validation_split, batch_size)

        # Need to add /usr/local/bin to PATH for local testing on MacOS
        self._env = os.environ.copy()
        self._env['PATH'] = '/usr/local/bin:' + self._env['PATH']

    def _get_image_data(self, filename):
        cmd = 'convert ' + filename + ' ' + c.IMAGE_MAGICK_FLAGS + ' png:-'
        raw_img = subprocess.check_output(cmd, env=self._env, shell=True)
        return Image.open(StringIO.StringIO(raw_img))
