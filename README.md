Convolutional Kanji
====

[TensorFlow](https://www.tensorflow.org/) implementation of a Convolutional Neural Network that recognizes handwritten Chinese characters.


### About

Convolution Neural Network trained to identify 100 different handwritten Chinese characters. Requires TensorFlow, NumPy and SciPy to be installed (see `requirements.txt`).

### Network architecture

1x40x40IN - 32C3 - MP2 - 64C3 - MP2 - 128C3 - MP2 - 512N - 100OUT


This represents a network with 40x40 grayscale image input,
a convolutional layer with 32 maps and 3x3 kernels, followed by max-pooling layer over non-overlapping regions of size 2x2,
a convolutional layer with 64 maps and 3x3 kernels, followed by max-pooling layer over non-overlapping regions of size 2x2,
a convolutional layer with 128 maps and 3x3 kernels, followed by max-pooling layer over non-overlapping regions of size 2x2,
a fully-connected layer with 512 hidden units and finally a fully-connected output layer with 100 units (one per class).

Dropout with probability 0.5 was used between the fully-connected hidden layer and the output layer.


### Performance

The CNN achieves an accuracy of 87% on the hold out test set.


### Dataset preparation

We assume that the original data is in the directory `original` and each class is in its own separate subdirectory, e.g. `original/character1`.

First, we resize the images to a given resolution (e.g. 32x32px), add white margin (e.g. 4px) around them to get a 40x40 matrix per image, and pickle each class into `pickled40` dir:

        python prepare_images.py original pickled40 32 4

Next, we randomly split the data into training, validation and test sets. The number of example for validation and test can be adjusted inside `prepare_datasets.py`. This produces an output called `data40_100.pickle` and a metadata file in `metadata/metadata.pickle`. The metadata contains information about image size, label map and mean image that will be needed for evaluation. The last argument to `prepare_datasets.py` is the number of classes to take, if empty take the full 100. 

        python prepare_datasets.py pickled40 data40_100.pickle

### Training

For training we use `train_cnn.py` which dataset from the previous step (e.g. `data40_100.pickle`) and starts optimizing the loss. Four times per each training epoch the model is saved into `models` dir in format `cnn_EPOCH_STEP.tf`.

        python prepare_datasets.py pickled40 data40_100.pickle

If the validation accuracy does not improve in the beginning after a few epochs, the optimizer's learning rate needs to be adjusted.

### Evaluation

Evaluation is done using `evaluate.py`. It takes a variable number of input images in PNG format. To run the script we require the `metadata/metadata.pickle` file to read in the parameters necessary for image preprocessing as well as the TensorFlow model in `models/cnn.tf`. `metadata/labels_unicode.txt` is needed to see unicode character printed, otherwise just the class code name (e.g. B0C1) is printed.

        python evaluate.py test/B0C1/image1.png test/B0DC/image2.png

Example output:
        
        test/B0C1/image1.png > B0C1 傲
        test/B0DC/image2.png > B0DC 败
