from __future__ import print_function

import csv

import numpy as np
from keras.preprocessing import image

from .constants import LOCAL_TEST, PREPROCESS_DIR, PREPROCESS_IMG_DIR


def load_delim_txt(fid, delim, fieldnames):
    f = open(fid, 'r')
    reader = csv.DictReader(f, delimiter=delim, fieldnames=fieldnames)
    d = []
    for row in reader:
        d += [row]
    f.close()
    return d


class SimpleLoader:

    def __init__(self):
        prefix = '../' if LOCAL_TEST else ''

        labelled = load_delim_txt(prefix + PREPROCESS_DIR + '/metadata/image_labels.txt',
                                  ' ', ('img', 'label'))

        self.imgs = []
        self.labels = []
        for pair in labelled:
            loaded = image.load_img(prefix + PREPROCESS_IMG_DIR + '/' + pair['img'], target_size=(224, 224))
            # Loads the image with the correct dim_ordering for the backend by default.
            loaded = image.img_to_array(loaded)
            self.imgs.append(loaded)
            self.labels.append(pair['label'])

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
