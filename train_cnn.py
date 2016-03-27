# -*- coding: UTF-8 -*-
import sys
import os
import pickle
import numpy as np
import tensorflow as tf

BATCH_SIZE = 45
KERNEL_SIZE = 5
CHANNELS = 1
DEPTH = 32
HIDDEN_NUM = 64
EPOCHS = 5000


def train(datasets, params):
    train, train_lbl = datasets['train'], datasets['train_lbl']
    valid, valid_lbl = datasets['valid'], datasets['valid_lbl']

    graph, x, y_, n_examples, preds, loss = params

    with tf.Session(graph=graph) as session:
        # create the optimizer to minimize the loss
        #optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        session.run(tf.initialize_all_variables())
        for step in range(EPOCHS):
            offset = (step * BATCH_SIZE) % (train_lbl.shape[0] - BATCH_SIZE)
            batch_data = train[offset:(offset+BATCH_SIZE), :, :, :]
            batch_labels = train_lbl[offset:(offset+BATCH_SIZE), :]

            feed_dict={x: batch_data, y_: batch_labels, n_examples: BATCH_SIZE}
            optimizer.run(feed_dict=feed_dict)

            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                print("step %d, training accuracy %g" % (step, train_accuracy))
                valid_accuracy = accuracy.eval(
                    feed_dict={x: valid, y_: valid_lbl, n_examples: len(valid)})
                print("step %d, valid accuracy %g" % (step, valid_accuracy))

    print 'Done'


def build_model(size, nlabels):

    def weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    graph = tf.Graph()
    with graph.as_default():
        # placeholders for input, None means batch of any size
        x = tf.placeholder(tf.float32, shape=(None, size, size, CHANNELS))
        y_ = tf.placeholder(tf.float32, shape=(None, nlabels))

        # create weights and biases for the 2 conv layers
        layer1_weights = weight([KERNEL_SIZE, KERNEL_SIZE, CHANNELS, DEPTH]) 
        layer1_biases = bias([DEPTH])

        layer2_weights = weight([KERNEL_SIZE, KERNEL_SIZE, DEPTH, DEPTH])
        layer2_biases = bias([DEPTH])

        layer3_weights = weight([size // 4 * size // 4 * DEPTH, HIDDEN_NUM])
        layer3_biases = bias([HIDDEN_NUM])

        layer4_weights = weight([HIDDEN_NUM, nlabels])
        layer4_biases = bias([nlabels])

        # connections between 2 conv layers and 1 fully-connected layer, using relus.
        conv = tf.nn.conv2d(x, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)

        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)

        shape = hidden.get_shape().as_list()
        n_examples = tf.placeholder(tf.int32)
        reshape = tf.reshape(hidden, tf.pack([n_examples, -1]))
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        logits = tf.matmul(hidden, layer4_weights) + layer4_biases

        # predict from logits using softmax
        predictions = tf.nn.softmax(logits)

        # cross-entropy as the loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))


        return (graph, x, y_, n_examples, predictions, loss)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python {} datasets.pickle".format(sys.argv[0])
        print """
Load the pickled datasets, train and evaluate the convolutional neural network,
then pickle and save it."""
        sys.exit(1)

    try:
        datasets = None
        with open(sys.argv[1], 'rb') as f:
            datasets = pickle.load(f)
        if len(datasets['train']) == 0:
            raise ValueError
    except IOError:
        print 'Failed to load dataset {}'.format(sys.argv[1])
    except ValueError:
        print 'Empty train dataset {}'.format(sys.argv[1])

    sx, sy, _ = datasets['train'][0].shape
    if sx != sy:
        raise ValueError('Input train data not a square')

    nlabels = len(datasets['label_map'])
    model_params = build_model(sx, nlabels)
    train(datasets, model_params)
