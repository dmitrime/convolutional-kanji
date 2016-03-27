# -*- coding: UTF-8 -*-
import sys
import os
import pickle
import numpy as np

# Validation and Test set fraction out of total number of examples.
VALID_PERCENT = 0.05
TEST_PERCENT = 0.05
RANDOM_SEED = 807


def reshape(data, lbl, nlabels):
    n = data[0].shape[0]
    data = data.reshape((-1, n*n)).astype(np.float32)
    lbl = (np.arange(nlabels) == lbl[:, None]).astype(np.float32)
    return data, lbl


def randomize(data, lbl):
    ps = np.random.permutation(len(data))
    data = data[ps]
    lbl = lbl[ps]
    return data, lbl

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python {} input-dir".format(sys.argv[0])
        print """
Load pickle files from input dir, random shuffle examples,\n\
and create a pickle containing training, validation and test sets and their labels."""
        sys.exit(1)
    inputdir = sys.argv[1]

    np.random.seed(RANDOM_SEED)

    # list all pickle files
    pkls = [f for f in os.listdir(inputdir) if f.endswith('.pickle')
            and os.path.isfile(os.path.join(inputdir, f))]
    num_classes = len(pkls)

    # we build our valid and test sets by taking a random fraction
    # of examples from each class, everything else is for training
    train, valid, test = [], [], []
    train_lbl, valid_lbl, test_lbl = [], [], []
    label_map = dict()
    for idx, pkl in enumerate(pkls):
        label = idx + 1
        label_map[label] = os.path.splitext(pkl)[0]
        path = os.path.join(inputdir, pkl)
        with open(path, 'rb') as f:
            class_data = pickle.load(f)
            # shuffle the class data
            np.random.shuffle(class_data)
            nexamples = len(class_data)

            vn = int(VALID_PERCENT*nexamples)
            valid.extend(class_data[:vn])
            valid_lbl.extend([label]*vn)

            tn = int(TEST_PERCENT*nexamples)
            test.extend(class_data[vn:vn+tn])
            test_lbl.extend([label]*tn)

            train.extend(class_data[vn+tn:])
            train_lbl.extend([label]*(nexamples-vn-tn))
        print '{}: finished processing {} exmaples.'.format(path, nexamples)

    # random shuffle the examples and labels in each set
    train, train_lbl = randomize(np.array(train), np.array(train_lbl))
    valid, valid_lbl = randomize(np.array(valid), np.array(valid_lbl))
    test, test_lbl = randomize(np.array(test), np.array(test_lbl))

    # reshape the examples into 1D arrays and labels into 1-hot encodings
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
        'label_map': label_map
    }

    try:
        with open('datasets.pickle', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except:
        print 'Failed to write pickled datasets...'

    print
    print 'Train: {}'.format(len(train))
    print 'Valid: {}'.format(len(valid))
    print 'Test: {}'.format(len(test))
    print 'Classes: {}'.format(len(label_map))
