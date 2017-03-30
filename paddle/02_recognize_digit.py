#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



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


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, dropout):
    weights = {

        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, 10]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([10]))
    }

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


def softmax_model(x):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    return y


def multilayer_perceptron(x):
    weights = {
        'h1': tf.Variable(tf.random_normal([784, 256])),
        'h2': tf.Variable(tf.random_normal([256, 256])),
        'out': tf.Variable(tf.random_normal([256, 10]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([256])),
        'b2': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([10]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


def main(_):
    n_input = 784
    n_classes = 10
    mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

    dropout = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # pred = softmax_model(x)
    # pred = multilayer_perceptron(x)
    pred = conv_net(x, dropout)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              dropout: 0.75})
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout: 1}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/mnist/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
