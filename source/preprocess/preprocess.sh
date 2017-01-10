#!/bin/bash
# Resizes all the training images to the same size and 
# saves them to PNG format using ImageMagick.
#
# Tasks:
# - resize the images
# - save to PNG format
#
# In addition, a label file at the image level is generated
# using information from the exams metadata table (see generate_labels.py).
#
# Author: Thomas Schaffter (thomas.schaff...@gmail.com)
# Last update: 2017-01-09

if [ "$LOCAL_TEST" == "1" ]; then
	PREFIX='../..'
else
	PREFIX=''
fi

# The default sed utility on macOS supports different command line options than the ones we need.
# You must install gnu-sed using Homebrew on macOS.
if [ $(uname) == "Darwin" ]; then
	SED='gsed'
else
	SED='sed'
fi

IMAGES_DIRECTORY="${PREFIX}/trainingData"
EXAMS_METADATA_FILENAME="${PREFIX}/metadata/exams_metadata.tsv"
IMAGES_CROSSWALK_FILENAME="${PREFIX}/metadata/images_crosswalk.tsv"

PREPROCESS_DIRECTORY="${PREFIX}/preprocessedData"
PREPROCESS_IMAGES_DIRECTORY="$PREPROCESS_DIRECTORY/images"
IMAGE_LABELS_FILENAME="$PREPROCESS_DIRECTORY/metadata/image_labels.txt"

echo "image labels" $IMAGE_LABELS_FILENAME

mkdir -p $PREPROCESS_IMAGES_DIRECTORY

echo "Resizing and converting $(find $IMAGES_DIRECTORY -name "*.dcm" | wc -l) DICOM images to PNG format"
find $IMAGES_DIRECTORY/ -name "*.dcm" | parallel --will-cite "convert {} -resize 224x224! $PREPROCESS_IMAGES_DIRECTORY/{/.}.png" # faster than mogrify
echo "PNG images have been successfully saved to $PREPROCESS_IMAGES_DIRECTORY/."

echo "Generating image labels to $IMAGE_LABELS_FILENAME"
python generate_image_labels.py $EXAMS_METADATA_FILENAME $IMAGES_CROSSWALK_FILENAME $IMAGE_LABELS_FILENAME
# Replace the .dcm extension to .png
${SED} -i 's/.dcm/.png/g' $IMAGE_LABELS_FILENAME

echo "Done"
