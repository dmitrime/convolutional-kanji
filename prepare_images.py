# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
from time import time

import utils

IMG_SIZE = 32
MARGIN_SIZE = 4


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python {} input-dir output-dir [image-size] [margin-size]".format(sys.argv[0])
        print """
Lists images in all subdirs of the input dir,\n\
resizes them to a square of given size (default: {}),\n\
then pickles the resized numpy matrices, add white margins (default: {})\n\
and writes them to output dir, one pickle file per subdir, in subdir-name.pkl format.\n\
Resolution of each image will be (size+margin, size+margin). """.format(IMG_SIZE, MARGIN_SIZE)
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

    if len(sys.argv) == 5:
        try:
            n = int(sys.argv[4])
            if n > 8 or n < 0:
                print 'margin size must be between 0 and 8!'
                sys.exit(1)
            MARGIN_SIZE = n
        except ValueError:
            print 'margin size must be an integer!'
            sys.exit(1)

    utils.ensure_ouputdir(outputdir)

    for d in os.listdir(inputdir):
        path = os.path.join(inputdir, d)
        if not os.path.isdir(path):
            continue

        pngs = [fn for fn in os.listdir(path) if fn.endswith('.png')]
        data = np.ndarray(shape=(len(pngs),
                          IMG_SIZE+2*MARGIN_SIZE,
                          IMG_SIZE+2*MARGIN_SIZE))
        start = time()
        for idx, png in enumerate(pngs):
            resized = utils.read_resize_image(os.path.join(path, png), IMG_SIZE)
            data[idx, :, :] = utils.add_margins(resized, MARGIN_SIZE)
        end = time()

        print '{}: {} images read and resized to {} in {:.3f}s. Saving...'.format(
            path, len(data), data[0].shape, end-start)

        utils.save_pickle({'data': data,
                           'image_size': IMG_SIZE,
                           'margin_size': MARGIN_SIZE},
                          d + '.pickle', outputdir)
