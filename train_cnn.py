# -*- coding: UTF-8 -*-
import os
import sys
import pickle
import tensorflow as tf

import utils

BATCH_SIZE = 100
EPOCHS = 20
MODEL_DIR = 'models'


def train(datasets):
    train, train_lbl = datasets['train'], datasets['train_lbl']
    valid, valid_lbl = datasets['valid'], datasets['valid_lbl']

    size = train[0].shape[0]
    nlabels = train_lbl.shape[1]

    # placeholders for input, None means batch of any size
    X = tf.placeholder(tf.float32, shape=(None, size, size, 1), name="X")
    y_ = tf.placeholder(tf.float32, shape=(None, nlabels), name="y_")
    keep = tf.placeholder(tf.float32)

    logits = build_model(X, nlabels, keep)
    # predict from logits using softmax
    predictions = tf.nn.softmax(logits)
    # cross-entropy as the loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
    # create the optimizer to minimize the loss
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

    utils.ensure_dir(MODEL_DIR)
    saver = tf.train.Saver()
    with tf.Session() as session:
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        session.run(tf.initialize_all_variables())

        nbatches = train_lbl.shape[0] / BATCH_SIZE
        for epoch in range(EPOCHS):
            for step in range(nbatches):
                batch_data = train[step*BATCH_SIZE:(step+1)*BATCH_SIZE, :, :, :]
                batch_labels = train_lbl[step*BATCH_SIZE:(step+1)*BATCH_SIZE, :]

                feed_dict = {X: batch_data, y_: batch_labels, keep: 0.5}
                optimizer.run(feed_dict=feed_dict)

                #print 'Batch labels:\n', batch_labels
                #print 'Predictions:\n', preds.eval(feed_dict=feed_dict)
                #print 'Correct pred:\n', correct_prediction.eval(feed_dict=feed_dict)

                if step % int(nbatches / 4) == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={X: batch_data, y_: batch_labels, keep: 1.0})
                    print("epoch %d, step %d, training accuracy %g" % (epoch+1, step, train_accuracy))

                    valid_accuracy = accuracy.eval(
                        feed_dict={X: valid, y_: valid_lbl, keep: 1.0})
                    print("epoch %d, step %d, valid accuracy %g" % (epoch+1, step, valid_accuracy))

                    # save the model
                    saver.save(session,
                       os.path.join(MODEL_DIR, 'cnn{}_e{}_s{}.tf'.format(valid[0].shape[0], epoch+1, step)))


def build_model(X, nlabels, keep):

    def weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # create weights and biases for the 1st conv layer
    l1_maps = 32
    l1_kernel = 3
    l1_weights = weight([l1_kernel, l1_kernel, 1, l1_maps])
    l1_biases = bias([l1_maps])

    # convolve input with the first kernels and apply relu
    h_conv1 = tf.nn.relu(conv2d(X, l1_weights) + l1_biases)
    # max-pool
    h_pool1 = max_pool(h_conv1)
    # apply dropout
    #h_pool1 = tf.nn.dropout(h_pool1, keep)

    # create weights and biases for the 2nd conv layer
    l2_maps = 64
    l2_kernel = 3
    l2_weights = weight([l2_kernel, l2_kernel, l1_maps, l2_maps])
    l2_biases = bias([l2_maps])

    # convolve first hidden layer output with the second kernels and apply relu
    h_conv2 = tf.nn.relu(conv2d(h_pool1, l2_weights) + l2_biases)
    # max-pool
    h_pool2 = max_pool(h_conv2)
    # apply dropout
    #h_pool2 = tf.nn.dropout(h_pool2, keep)

    # create weights and biases for the 2nd conv layer
    l3_maps = 128
    l3_kernel = 3
    l3_weights = weight([l3_kernel, l3_kernel, l2_maps, l3_maps])
    l3_biases = bias([l3_maps])

    # convolve first hidden layer output with the second kernels and apply relu
    h_conv3 = tf.nn.relu(conv2d(h_pool2, l3_weights) + l3_biases)
    # max-pool
    h_pool3 = max_pool(h_conv3)
    # apply dropout
    #h_pool3 = tf.nn.dropout(h_pool3, keep)

    # create weights and biases for the 3rd hidden layer
    hidden_num = 512
    dim = h_pool3.get_shape().as_list()
    l4_weights = weight([dim[1]*dim[2]*dim[3], hidden_num])
    l4_biases = bias([hidden_num])

    # fully connect the 2nd layer output to 3rd input
    dim = h_pool3.get_shape().as_list()
    reshape = tf.reshape(h_pool3, [-1, dim[1]*dim[2]*dim[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, l4_weights) + l4_biases)
    # apply dropout
    hidden = tf.nn.dropout(hidden, keep)

    # create weights and biases for the output layer
    l5_weights = weight([hidden_num, nlabels])
    l5_biases = bias([nlabels])

    return tf.matmul(hidden, l5_weights) + l5_biases


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

    train(datasets)
