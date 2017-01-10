# Makefile to test training locally
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

all: test-local-simple

docker-build:
	-docker rm -f `docker ps -a -q  --filter name=$(DOCKER_NAME)` 2> /dev/null || true
	cd .. && docker build -f Dockerfile-train -t $(DOCKER_REPO) .
	-docker rmi $$(docker images -f 'dangling=true' -q) 2> /dev/null || true

docker-push: docker-build
	docker push $(DOCKER_REPO)

test-prep:
	echo "Creating directory to store model state..."
	mkdir -p $(MODELSTATE_DIR)

test-local-simple: test-prep
	source ../../pyenv/bin/activate && env LOCAL_TEST=1 python ../train_mammo_vgg.py

test-local: test-prep docker-build
	docker run -d -P --name csc490-training \
		   -v $(TRAINING_DATA_DIR):/trainingData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(PREPROCESS_DIR):/preprocessedData \
		   -v $(MODELSTATE_DIR):/modelState \
		   $(DOCKER_REPO) /train.sh
