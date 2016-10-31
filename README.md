# Mammogram Classifier

Currently there is a limiter on the  number  of imgs loaded,  since the dicom  format is quite obese.
A good approach to reducing the  footprint of the data might  be to do some severe dimensionality reduction and
feature extraction at the img loading stage.

### Data

**First** download the data from https://www.synapse.org/#!Synapse:syn4224222/wiki/401757
to the data dir

**Next** Ungzip all dcm files (this should be automated into  the pipeline...

### Dependencies
	- pydicom
	- numpy

### TODO
Functionality for dataset inflation  (add reflected images, images with noise added, rotated images).  
Image preprocessing.  



