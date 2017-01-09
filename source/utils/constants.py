# Global constant definitions

import os

# Indicates whether we're in a local test environment or not.
LOCAL_TEST = os.getenv('LOCAL_TEST', None) is not None

DIR_PREFIX = '../..' if LOCAL_TEST else ''

# Directory where raw training images are stored according to challenge guidelines.
RAW_IMG_DIR = DIR_PREFIX + '/trainingData'

# Directory where preprocessed data can be saved according to challenge guidelines.
PREPROCESS_DIR = DIR_PREFIX + '/preprocessedData'

# Directory where preprocessed images are stored.
PREPROCESS_IMG_DIR = PREPROCESS_DIR + '/images'

# Location where exams_metadata.tsv is stored according to challenge guidelines.
EXAMS_METADATA_FILENAME = DIR_PREFIX + '/metadata/exams_metadata.tsv'
# Location where images_crosswalk.tsv is stored according to challenge guidelines.
IMAGES_CROSSWALK_FILENAME = DIR_PREFIX + '/metadata/images_crosswalk.tsv'

# Location to save the trained model according to challenge guidelines.
MODELSTATE_DIR = DIR_PREFIX + '/modelState'
