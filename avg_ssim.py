# -*- coding: UTF-8 -*-
import sys
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from skimage.measure import structural_similarity as ssim

import utils

RANDOM_SEED = 807

def plot(res):
    fig, ax = plt.subplots()
    ax.bar(range(len(res)), res, alpha=0.5)
    ax.set_ylabel('Average SSIM')
    ax.set_xlabel('Class number') 
    ax.set_title('Average SSIM distribution for character classes') 
    ax.axvline(3, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(97, color='r', linestyle='dashed', linewidth=2)
    plt.show()

def avg_ssim(data, N=1000):
    allpairs = [comb for comb in combinations(range(len(data)), 2)]
    random.shuffle(allpairs)
    return np.mean([ssim(data[a], data[b]) for a, b in allpairs[:N]])

if __name__ == '__main__':
    ssimfile = 'ssims.in'
    # compute or plot
    if False:
        if len(sys.argv) < 2:
            print "Usage: python {} input-dir".format(sys.argv[0])
            sys.exit(1)

        inputdir = sys.argv[1]
        np.random.seed(RANDOM_SEED)

        # list all pickle files
        pkls = [f for f in os.listdir(inputdir) if f.endswith('.pickle')
                and os.path.isfile(os.path.join(inputdir, f))]

        # we build our valid and test sets by taking a random fraction
        # of examples from each class, everything else is for training
        label_map = dict()
        image_size, margin_size = 0, 0
        ssims = []
        for label, pkl in enumerate(pkls):
            label_map[label] = os.path.splitext(pkl)[0]
            path = os.path.join(inputdir, pkl)
            assim = None
            with open(path, 'rb') as f:
                data = pickle.load(f)
                class_data = data['data']
                image_size = data['image_size']
                margin_size = data['margin_size']

                ssims.append((avg_ssim(class_data),
                             pkl.split('.')[0]))

            print '{}: avg ssim: {}'.format(path, ssims[-1][0])

        with open(ssimfile, 'w') as g:
            for assim, lbl in sorted(ssims):
                g.write("{} {}\n".format(assim, lbl))
    else:
        res = []
        with open(ssimfile, 'r') as f:
            for line in f:
                res.append(float(line.split()[0]))
        plot(res)
