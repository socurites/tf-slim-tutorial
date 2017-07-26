__author__ = 'socurites@gmail.com'

"""
VGG-16 network definition example by TF-Slim
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
VGG-16 Network
Ref: https://www.cs.toronto.edu/~frossard/post/vgg16/
"""
# FIG.2 - Macro Architecture of VGG16

# VGG-16 network definition by native-TF


"""
VGG-16 network definition by TF-Slim
# TF-Slim model: https://github.com/tensorflow/models/tree/master/slim/nets
# TF-Slim VGG model: https://github.com/tensorflow/models/blob/master/slim/nets/vgg.py
"""
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net

# import pre-defined network model of TF-Slim
import tensorflow.contrib.slim.nets
vgg = tf.contrib.slim.nets.vgg