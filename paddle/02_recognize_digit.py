#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')



def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)


def softmax_regression(features, labels):
    W = tf.get_variable("W", [784, 10], dtype=tf.float64)
    b = tf.get_variable("b", [10], dtype=tf.float64)
    y = features['x'] * W + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(predictions=y, loss=loss, train_op=train)


def multilayer_perceptron():
    pass


# Create model
def conv_net(x, weights, biases, dropout):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# def convolutional_neural_network():
#     # Hidden 1
#     with tf.name_scope('hidden1'):
#         weights = tf.Variable(
#             tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
#                                 stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
#             name='weights')
#         biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
#         hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
#     # Hidden 2
#     with tf.name_scope('hidden2'):
#         weights = tf.Variable(
#             tf.truncated_normal([hidden1_units, hidden2_units],
#                                 stddev=1.0 / math.sqrt(float(hidden1_units))),
#             name='weights')
#         biases = tf.Variable(tf.zeros([hidden2_units]),
#                              name='biases')
#         hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
#     # Linear
#     with tf.name_scope('softmax_linear'):
#         weights = tf.Variable(
#             tf.truncated_normal([hidden2_units, NUM_CLASSES],
#                                 stddev=1.0 / math.sqrt(float(hidden2_units))),
#             name='weights')
#         biases = tf.Variable(tf.zeros([NUM_CLASSES]),
#                              name='biases')
#         logits = tf.matmul(hidden2, weights) + biases
#     return logits
#     pass


def input_fn(dataset, batch_size):
    batch_xs, label = dataset.next_batch(batch_size)
    features = {"x": batch_xs}
    return features, label


def main(_):
    mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)
    estimator = tf.contrib.learn.Estimator(model_fn=softmax_regression)
    # define our data set

    # train
    estimator.fit(input_fn=lambda: input_fn(mnist.train, 100), steps=100)
    # evaluate our model
    print(estimator.evaluate(input_fn=lambda: input_fn(mnist.test, 100), steps=10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/mnist/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
