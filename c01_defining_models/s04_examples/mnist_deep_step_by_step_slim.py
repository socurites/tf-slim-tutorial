__author__ = 'socurites@gmail.com'

"""
MNIST network definition example by TF-Slim
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mnist_convnet(inputs, is_training=True):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1)):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=5):
            net = slim.conv2d(inputs=inputs, num_outputs=32, scope='conv1')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            net = slim.conv2d(inputs=net, num_outputs=64, scope='conv2')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            net = slim.flatten(inputs=net, scope='flatten')
            net = slim.fully_connected(inputs=net, num_outputs=1024, scope='fc3')
            net = slim.dropout(inputs=net, is_training=is_training, keep_prob=0.5, scope='dropout4')
            net = slim.fully_connected(inputs=net, num_outputs=10, activation_fn=None, scope='fc4')
    return net
