import os
import pickle
import numpy as np
import tensorflow as tf

from train_cnn import build_model, MODEL_DIR
from prepare_datasets import METADATA_DIR, METADATA_FILE

import matplotlib.pyplot as plt

MODEL_NAME = 'cnn.tf'

def gauss(x):
    return np.random.normal(scale=x, size=(40,40,1))

def plot(xs, ys):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, [100.0 * y for y in ys])

    plt.axvline(5.0, ymax=0.855, color='r', linestyle='dashed', linewidth=2)
    plt.axhline(88.5, xmax=1./5.+0.05, color='r', linestyle='dashed', linewidth=2)

    plt.xticks(np.arange(min(xs), max(xs)+1, 1.0))
    yticks = plt.FormatStrFormatter('%.0f%%')
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel('Standard deviation of noise')
    ax.set_ylabel('Accuracy')
    plt.title('Robustness to random noise')
    plt.show()

if __name__ == '__main__':

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

    X = tf.placeholder(tf.float32, shape=(None, size, size, 1), name="X")
    keep = tf.placeholder(tf.float32)

    # make predictions
    predictions = tf.nn.softmax(build_model(X, nlabels, keep))

    import pickle
    with open('d40_100.pickle', 'rb') as f:
        d = pickle.load(f)
    y_ = tf.placeholder(tf.float32, shape=(None, nlabels), name="y_")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model):
        raise Exception('Model {} does not exist!'.format(model))

    #res = []
    #saver = tf.train.Saver()
    #with tf.Session() as session:
        #saver.restore(session, model)
        #test_accuracy = accuracy.eval(feed_dict={y_: d['test_lbl'], X: d['test'], keep: 1.0})
        #print 'Test: {}'.format(test_accuracy)
        #res.append(test_accuracy)

    #for ran in np.arange(1.0, 21.0, 1.0):
        #tset = np.ndarray(shape=(1000, 40, 40, 1))
        #for i in range(len(d['test'])):
            #tset[i] = d['test'][i] + gauss(ran)

        #with tf.Session() as session:
            #saver.restore(session, model)
            #test_accuracy = accuracy.eval(feed_dict={y_: d['test_lbl'], X: tset, keep: 1.0})
            #print 'Test: {}'.format(test_accuracy)
            #res.append(test_accuracy)

    #with open('robustness.in', 'w') as f:
        #for step, acc in zip(np.arange(0.0, 21.0, 1.0), res):
            #f.write('{} {}\n'.format(step, acc))
    
    xs, ys = [], []
    with open('robustness.in', 'r') as f:
        for line in f:
            x,y = line.split()
            xs.append(float(x))
            ys.append(float(y))
    print xs, ys
    plot(xs, ys)
