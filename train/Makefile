# Makefile to test training locally
# Preconditions:
#     You must have docker installed
#     ../preprocessedData must exist and contain PNG files of all the images
#     ../preprocessedData/metadata must exist and contain image_labels.txt

TRAINING_DATA_DIR := ${CURDIR}/../trainingData
METADATA_DIR := ${CURDIR}/../metadata
PREPROCESS_DIR := ${CURDIR}/../preprocessedData
DOCKER_TAG := csc490-train-local

all: test_local_simple

docker-build:
	docker build -t $(DOCKER_TAG) .

test_local_simple:
	env LOCAL_TEST=1 python train_mammo_vgg.py

test_local: docker-build
	docker run -d -P --name csc490-training \
		   -v $(TRAINING_DATA_DIR):/trainingData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(PREPROCESS_DIR):/preprocessedData \
		   $(DOCKER_TAG) /train.sh
