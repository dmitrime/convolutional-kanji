# -*- coding: UTF-8 -*-
import sys
import pickle
import numpy as np
import tensorflow as tf

import utils
from train_cnn import build_model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python {} image1.png image2.png ...".format(sys.argv[0])
        print """
Predict the Chinese character for the given images."""
        sys.exit(1)

    # TODO
    # read in default params for size and marging
    img_size = 32
    margin_size = 4
    size = img_size + margin_size*2

    with open('meta.pickle', 'rb') as f:
        meta = pickle.load(f)
    label_map = meta['label_map']
    nlabels = len(label_map)
    mean_image = meta['mean_image']

    with open('original/labels_unicode.txt', 'r') as f:
        labels_unicode = {k: v for k,v in [line.strip().split() for line in f]}
    # prettify the above
    # TODO

    images = sys.argv[1:]
    data = np.ndarray(shape=(len(images), size, size, 1))
    for idx, img in enumerate(images):
        data[idx, :, :, 0] = utils.add_margins(
            utils.read_resize_image(img, img_size), margin_size) - mean_image


    X = tf.placeholder(tf.float32, shape=(None, size, size, 1), name="X")
    keep = tf.placeholder(tf.float32)

    # make predictions
    predictions = tf.nn.softmax(build_model(X, nlabels, keep))

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, 'models/good/cnn40_e15_s0.tf')
        print 'session loaded'
        preds = session.run(predictions, feed_dict={X: data, keep: 1.0})
        classes = np.argmax(preds, 1)
        for img, c in zip(images, classes):
            print '{} > {} {}'.format(img, label_map[c], labels_unicode.get(label_map[c]))
