# -*- coding: UTF-8 -*-
import os
import sys
import pickle
import numpy as np
import tensorflow as tf

import utils
from train_cnn import build_model, MODEL_DIR
from prepare_datasets import METADATA_DIR, METADATA_FILE

LABEL_UNICODE = 'labels_unicode.txt'
MODEL_NAME = 'cnn.tf'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python {} image1.png image2.png ...".format(sys.argv[0])
        print """
Predict the Chinese character for the given images."""
        sys.exit(1)

    # read in params for label map, mean image and sizes
    mf = os.path.join(METADATA_DIR, METADATA_FILE)
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            meta = pickle.load(f)
            label_map = meta['label_map']
            mean_image = meta['mean_image']
            img_size = meta['image_size']
            margin_size = meta['margin_size']
    else:
        raise Exception('metadata file {} does not exist!'.format(meta))

    size = img_size + margin_size*2
    nlabels = len(label_map)

    # read in label unicode map
    lu = os.path.join(METADATA_DIR, LABEL_UNICODE)
    labels_unicode = dict()
    if os.path.exists(lu):
        with open(lu, 'r') as f:
            labels_unicode = {k: v for k,v in [line.strip().split() for line in f]}

    images = sys.argv[1:]
    data = np.ndarray(shape=(len(images), size, size, 1))
    for idx, img in enumerate(images):
        data[idx, :, :, 0] = utils.add_margins(
            utils.read_resize_image(img, img_size), margin_size) - mean_image


    X = tf.placeholder(tf.float32, shape=(None, size, size, 1), name="X")
    keep = tf.placeholder(tf.float32)

    # make predictions
    predictions = tf.nn.softmax(build_model(X, nlabels, keep))

    model = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model):
        raise Exception('Model {} does not exist!'.format(model))

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, model)
        print 'session loaded'
        preds = session.run(predictions, feed_dict={X: data, keep: 1.0})
        classes = np.argmax(preds, 1)
        for img, c in zip(images, classes):
            print '{} > {} {}'.format(img, label_map[c], labels_unicode.get(label_map[c], ''))
