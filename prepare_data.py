# -*- coding: UTF-8 -*-
import sys
import os
import pickle
import numpy as np
from time import time
from scipy import ndimage
from scipy.misc import imresize, toimage

IMG_SIZE = 32

def save_pickle(data, name, od):
    """ Save all the resized images into one pickle file. """
    picklename = os.path.join(od, name + '.pickle')
    try:
        with open(picklename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except:
        print 'Failed to write data into {}'.format(picklename)
        

def read_resize_image(img):
    """ Read the image, resize it and return a numpy matrix of new size. """
    try:
        return imresize(ndimage.imread(img), (IMG_SIZE, IMG_SIZE))
    except IOError:
        print 'File {} not readable, skipping.'.format(img)

def ensure_ouputdir(od):
    """ Create the output dir if it does not already exist. """
    if not os.path.isdir(od):
        os.mkdir(od)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python {} input-dir output-dir [size]".format(sys.argv[0])
        print """
Lists images in all subdirs of the input dir,\n\
resizes them to a square of given size (default: {}),\n\
then pickles the resized numpy matrices and writes them to output dir,\n\
one pickle file per subdir, in subdir-name.pkl format.""".format(IMG_SIZE)
        sys.exit(1)

    inputdir, outputdir = sys.argv[1], sys.argv[2]

    if len(sys.argv) == 4:
        try:
            n = int(sys.argv[3])
            if n > 64 or n < 8:
                print 'size must be between 8 and 64!'
                sys.exit(1)
            IMG_SIZE = n
        except ValueError:
            print 'size must be an integer!'
            sys.exit(1)

    ensure_ouputdir(outputdir)

    for d in os.listdir(inputdir):
        path = os.path.join(inputdir, d)
        if not os.path.isdir(path):
            continue

        pngs = [fn for fn in os.listdir(path) if fn.endswith('.png')]
        data = np.ndarray(shape=(len(pngs), IMG_SIZE, IMG_SIZE))
        start = time()
        for idx, png in enumerate(pngs):
            data[idx, :, :] = read_resize_image(os.path.join(path, png))
        end = time()

        print '{}: {} images read and resized to {} in {:.3f}s. Saving...'.format(
            path, len(data), data[0].shape, end-start)

        save_pickle(data, d, outputdir)


