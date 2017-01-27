# Makefile to test scoring locally and to build docker images.
# Preconditions:
#     You must have docker installed
#     ../../preprocessedData must exist and contain PNG files of all the images
#     ../../preprocessedData/metadata must exist and contain image_labels.txt
#     ../../modelState must exist and must contain trained.h5 and weights.h5

INFERENCE_DATA_DIR := ${CURDIR}/../../inferenceData
METADATA_DIR := ${CURDIR}/../../metadata
MODELSTATE_DIR := ${CURDIR}/../../modelState
OUTPUT_DIR := ${CURDIR}/../../output
DOCKER_REPO := docker.synapse.org/syn7506720/csc490-score
DOCKER_NAME := csc490-scoring

all: test-local-simple

docker-build:
	-docker rm -f `docker ps -a -q --filter name=$(DOCKER_NAME)` 2> /dev/null || true
	cd .. && docker build -f Dockerfile-score -t $(DOCKER_REPO) .
	-docker rmi $$(docker images -f 'dangling=true' -q) 2> /dev/null || true

docker-push: docker-build
	docker push $(DOCKER_REPO)

test-prep:
	echo "Creating directory to store output scores..."
	mkdir -p $(OUTPUT_DIR)

test-local-simple: test-prep
	source ${CURDIR}/../../pyenv/bin/activate && env LOCAL_TEST=1 python ${CURDIR}/../score_mammo.py

test-local: test-prep docker-build
	docker run -d -P --name $(DOCKER_NAME) \
		   -v $(INFERENCE_DATA_DIR):/inferenceData:ro \
		   -v $(METADATA_DIR):/metadata:ro \
		   -v $(MODELSTATE_DIR):/modelState:ro \
		   -v $(OUTPUT_DIR):/output \
		   $(DOCKER_REPO) /sc1_infer.sh
