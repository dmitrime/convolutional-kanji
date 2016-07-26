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
#MODEL_NAME = 'good2/cnn40_e15_s0.tf'
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

    #import pickle
    #with open('d40_100.pickle', 'rb') as f:
        #d = pickle.load(f)
    #y_ = tf.placeholder(tf.float32, shape=(None, nlabels), name="y_")
    #correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model):
        raise Exception('Model {} does not exist!'.format(model))

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, model)
        #print 'session loaded'
        preds = session.run(predictions, feed_dict={X: data, keep: 1.0})
        classes = np.argmax(preds, 1)
        for img, c in zip(images, classes):
            print '{} > {} {}'.format(img, label_map[c], labels_unicode.get(label_map[c], ''))


        #feed_dict={y_: d['test_lbl'], X: d['test'], keep: 1.0}
        #test_accuracy = accuracy.eval(feed_dict=feed_dict)
        #is_correct = correct_prediction.eval(feed_dict=feed_dict)
        #misclassified_idxs = np.arange(len(d['test']))[~is_correct]

        #print 'Number of misclassified examples:', len(misclassified_idxs)

        #from collections import Counter
        #c = Counter()
        #for idx in misclassified_idxs:
            #val = d['test_lbl'][idx].nonzero()[0][0]
            #c[label_map[val]] += 1
        #for k, v in sorted(c.items(), reverse=True, key=lambda x: x[1]):
            #print k, v

        #print
        #print 'Full test set: {:.2f}% accuracy'.format(test_accuracy*100.0)
