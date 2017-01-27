# Makefile to test training locally and to build docker images.
# Preconditions:
#     You must have docker installed
#     ../../preprocessedData must exist and contain PNG files of all the images
#     ../../preprocessedData/metadata must exist and contain image_labels.txt

TRAINING_DATA_DIR := ${CURDIR}/../../trainingData
METADATA_DIR := ${CURDIR}/../../metadata
PREPROCESS_DIR := ${CURDIR}/../../preprocessedData
MODELSTATE_DIR := ${CURDIR}/../../modelState
DOCKER_REPO := docker.synapse.org/syn7506720/csc490-train
DOCKER_NAME := csc490-training
DOCKER_REPO_RELU := docker.synapse.org/syn7506720/csc490-train-relu
DOCKER_NAME_RELU := csc490-training-relu

all: test-local-simple

docker-build:
	-docker rm -f `docker ps -a -q --filter name=$(DOCKER_NAME)` 2> /dev/null || true
	cd .. && docker build -f Dockerfile-train -t $(DOCKER_REPO) .
	-docker rmi $$(docker images -f 'dangling=true' -q) 2> /dev/null || true

docker-push: docker-build
	docker push $(DOCKER_REPO)

test-prep:
	echo "Creating directory to store model state..."
	mkdir -p $(MODELSTATE_DIR)

test-local-simple: test-prep
	source ${CURDIR}/../../pyenv/bin/activate && env LOCAL_TEST=1 python ${CURDIR}/../train_mammo_vgg.py

test-local: test-prep docker-build
	docker run -d -P --name $(DOCKER_NAME) \
		   -v $(TRAINING_DATA_DIR):/trainingData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(PREPROCESS_DIR):/preprocessedData \
		   -v $(MODELSTATE_DIR):/modelState \
		   $(DOCKER_REPO) /train.sh

docker-build-relu:
	-docker rm -f `docker ps -a -q --filter name=$(DOCKER_NAME_RELU)` 2> /dev/null || true
	cd .. && docker build -f Dockerfile-train-relu -t $(DOCKER_REPO_RELU) .
	-docker rmi $$(docker images -f 'dangling=true' -q) 2> /dev/null || true

docker-push-relu: docker-build-relu
	docker push $(DOCKER_REPO_RELU)

test-local-simple-relu: test-prep
	source ${CURDIR}/../../pyenv/bin/activate && \
	       env USE_RELU=1 LOCAL_TEST=1 python ${CURDIR}/../train_mammo_vgg.py

test-local-relu: test-prep docker-build-relu
	docker run -d -P --name $(DOCKER_NAME_RELU) \
		   -v $(TRAINING_DATA_DIR):/trainingData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(PREPROCESS_DIR):/preprocessedData \
		   -v $(MODELSTATE_DIR):/modelState \
		   $(DOCKER_REPO_RELU) /train.sh
