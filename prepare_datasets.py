# -*- coding: UTF-8 -*-
import sys
import os
import pickle
import numpy as np

import utils

# Validation and Test set fraction out of total number of examples.
VALID_PER_CLASS = 10
TEST_PER_CLASS = 10
RANDOM_SEED = 807

METADATA_DIR = 'metadata'
METADATA_FILE = 'metadata.pickle'


def reshape(data, lbl, nlabels):
    n = data[0].shape[0]
    # reshape to add a single (grayscale) channel,
    # because tf.nn.conv2d() expects 4-D input.
    data = data.reshape((-1, n, n, 1)).astype(np.float32)
    # 1-hot encoding for the labels
    lbl = (np.arange(nlabels) == lbl[:, None]).astype(np.float32)
    return data, lbl


def randomize(data, lbl):
    ps = np.random.permutation(len(data))
    data = data[ps]
    lbl = lbl[ps]
    return data, lbl

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python {} input-dir output.pickle [nclasses]".format(sys.argv[0])
        print """
Load pickle files from input dir, random shuffle examples,\n\
and create a pickle containing training, validation and test sets and their labels.
If nclasses param is given, limit the dataset to N classes."""
        sys.exit(1)

    nclasses = None
    if len(sys.argv) == 4:
        try:
            n = int(sys.argv[3])
            if n <= 1 or n > 100:
                raise ValueError
            nclasses = n
        except:
            print 'Number of classes is between 2 and 100'
            sys.exit(1)
    inputdir = sys.argv[1]
    outfile = sys.argv[2]

    np.random.seed(RANDOM_SEED)

    # list all pickle files
    pkls = [f for f in os.listdir(inputdir) if f.endswith('.pickle')
            and os.path.isfile(os.path.join(inputdir, f))]

    # we build our valid and test sets by taking a random fraction
    # of examples from each class, everything else is for training
    train, valid, test = [], [], []
    train_lbl, valid_lbl, test_lbl = [], [], []
    label_map = dict()
    image_size, margin_size = 0, 0
    for label, pkl in enumerate(pkls):
        label_map[label] = os.path.splitext(pkl)[0]
        path = os.path.join(inputdir, pkl)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            class_data = data['data']
            image_size = data['image_size']
            margin_size = data['margin_size']
            # shuffle the class data
            np.random.shuffle(class_data)
            nexamples = len(class_data)

            vn = VALID_PER_CLASS
            valid.extend(class_data[:vn])
            valid_lbl.extend([label]*vn)

            tn = TEST_PER_CLASS
            test.extend(class_data[vn:vn+tn])
            test_lbl.extend([label]*tn)

            train.extend(class_data[vn+tn:])
            train_lbl.extend([label]*(nexamples-vn-tn))
        print '{}: finished processing {} exmaples.'.format(path, nexamples)

        if nclasses is not None and label+1 == nclasses:
            break

    # subtract mean image
    mean_image = np.mean(train, axis=0)
    train = np.array(train) - mean_image
    valid = np.array(valid) - mean_image
    test = np.array(test) - mean_image

    # random shuffle the examples and labels in each set
    train, train_lbl = randomize(train, np.array(train_lbl))
    valid, valid_lbl = randomize(valid, np.array(valid_lbl))
    test, test_lbl = randomize(test, np.array(test_lbl))

    # reshape the examples into 4-D arrays and labels into 1-hot encodings
    nlabels = len(label_map)
    train, train_lbl = reshape(train, train_lbl, nlabels)
    valid, valid_lbl = reshape(valid, valid_lbl, nlabels)
    test, test_lbl = reshape(test, test_lbl, nlabels)

    # store all in a dict and pickle it
    data = {
        'train': train,
        'train_lbl': train_lbl,
        'valid': valid,
        'valid_lbl': valid_lbl,
        'test': test,
        'test_lbl': test_lbl,
    }

    utils.save_pickle(data, outfile)

    print
    print 'Train: {}'.format(len(train))
    print 'Valid: {}'.format(len(valid))
    print 'Test: {}'.format(len(test))
    print 'Classes: {}'.format(len(label_map))
    print
    print 'Dataset written to {}'.format(outfile)

    # save label map, mean image and sizes
    meta = {
        'label_map': label_map,
        'mean_image': mean_image,
        'image_size': image_size,
        'margin_size': margin_size
    }
    utils.ensure_dir(METADATA_DIR)
    fm = os.path.join(METADATA_DIR, METADATA_FILE)
    utils.save_pickle(meta, fm)

    print 'Metadata written to {}'.format(fm)
