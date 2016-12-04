# Makefile to test preprocessing locally
# Preconditions:
#     You must have docker installed
#     ../trainingData directory must exist and contain all of the extracted DCM files
#     ../metadata directory must exist and contain exams_metadata.tsv and images_crosswalk.tsv

TRAINING_DATA_DIR := ${CURDIR}/../trainingData
METADATA_DIR := ${CURDIR}/../metadata
PREPROCESS_DIR := ${CURDIR}/../preprocessedData
DOCKER_REPO := docker.synapse.org/syn7506720/csc490-preprocess
DOCKER_NAME := csc490-preprocessing

all: test_local

docker-build:
	docker rm -f `docker ps -a -q  --filter name=$(DOCKER_NAME)`
	docker build -t $(DOCKER_REPO) .
	-docker rmi $(docker images -f 'dangling=true' -q)

docker-push: docker-build
	docker push $(DOCKER_REPO)

test-prep:
	echo "Creating directory to store preprocessed data..."
	mkdir -p $(PREPROCESS_DIR)

test_local: test-prep docker-build
	docker run -d -P --name csc490-preprocessing \
		   -v $(TRAINING_DATA_DIR):/trainingData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(PREPROCESS_DIR):/preprocessedData \
		   $(DOCKER_REPO) /preprocess.sh
