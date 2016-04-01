# -*- coding: UTF-8 -*-
import os
import pickle
import numpy as np
from scipy import ndimage
from scipy.misc import imresize


def add_margins(img, margin_size):
    """ Add N-pixel white margins to each image. """
    return np.pad(img, margin_size, 'constant', constant_values=255)


def save_pickle(data, name, od="."):
    """ Save all the resized images into one pickle file. """
    picklename = os.path.join(od, name)
    try:
        with open(picklename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except:
        print 'Failed to write data into {}'.format(picklename)


def read_resize_image(img, img_size):
    """ Read the image, resize it and return a numpy matrix of new size. """
    try:
        return imresize(ndimage.imread(img), (img_size, img_size))
    except IOError:
        print 'File {} not readable, skipping.'.format(img)


def ensure_dir(od):
    """ Create the output dir if it does not already exist. """
    if not os.path.isdir(od):
        os.mkdir(od)
