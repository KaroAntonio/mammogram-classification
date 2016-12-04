# Global constant definitions

import os

# Indicates whether we're in a local test environment or not.
LOCAL_TEST = os.getenv('LOCAL_TEST', None) is not None

# Directory where raw training images are stored according to challenge guidelines.
RAW_IMG_DIR = '/trainingData'

# Directory where preprocessed data can be saved according to challenge guidelines.
PREPROCESS_DIR = '/preprocessedData'

# Directory where preprocessed images are stored.
PREPROCESS_IMG_DIR = PREPROCESS_DIR + '/images'

# Location where exams_metadata.tsv is stored according to challenge guidelines.
EXAMS_METADATA_FILENAME = '/metadata/exams_metadata.tsv'
# Location where images_crosswalk.tsv is stored according to challenge guidelines.
IMAGES_CROSSWALK_FILENAME = '/metadata/images_crosswalk.tsv'
