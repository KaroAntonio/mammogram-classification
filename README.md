# Mammogram Classifier

Currently we convert the DICOM images into 8-bit RGB PNGs with dimensions 224x224.

### Data

**First** download the data from https://www.synapse.org/#!Synapse:syn4224222/wiki/401757
to trainingData/ dir in root of repository.

**Next** Ungzip all dcm files (this should be automated into the pipeline...) and setup the 
folder structure according to the specifications in the Makefiles.

### Dependencies
	- python 2.7
	- python libraries in requirements.txt (create a virtualenv in a folder named 'pyenv' in the
      root folder of the repo, activate it and 'pip install -r requirements.txt')
	- ImageMagick
	- parallel (aka gnu-parallel)
	- gnu-sed (only if you're running macOS)


### TODO
Functionality for dataset inflation  (add reflected images, images with noise added, rotated images).  
Image preprocessing.  

### RESOURCES
https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

