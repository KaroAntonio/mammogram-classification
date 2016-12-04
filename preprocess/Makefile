# Makefile to test preprocessing locally
# Preconditions:
#     You must have docker installed
#     ../trainingData directory must exist and contain all of the extracted DCM files
#     ../metadata directory must exist and contain exams_metadata.tsv and images_crosswalk.tsv

TRAINING_DATA_DIR := ${CURDIR}/../trainingData
METADATA_DIR := ${CURDIR}/../metadata
PREPROCESS_DIR := ${CURDIR}/../preprocessedData
DOCKER_TAG := csc490-preprocess-local
DOCKER_NAME := csc490-preprocessing

all: test_local

docker-build:
	docker rm -f `docker ps -a -q  --filter name=$(DOCKER_NAME)`
	docker build -t $(DOCKER_TAG) .
	-docker rmi $(docker images -f 'dangling=true' -q)

test-prep:
	echo "Creating directory to store preprocessed data..."
	mkdir -p $(PREPROCESS_DIR)

test_local: test-prep docker-build
	docker run -d -P --name csc490-preprocessing \
		   -v $(TRAINING_DATA_DIR):/trainingData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(PREPROCESS_DIR):/preprocessedData \
		   $(DOCKER_TAG) /preprocess.sh
