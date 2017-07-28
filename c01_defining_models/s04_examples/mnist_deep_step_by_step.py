# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
MNIST example by TF-native
from: https://github.com/socurites/StartingTensorflow
"""

import tensorflow as tf
import numpy as np

# Load MNIST datastes
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

# create session
sess = tf.InteractiveSession()

# create variable for input and real output y
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# reshape x
x_image = tf.reshape(x, [-1,28,28,1])

# Define weight(kernel filter) with shape
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Define bias with shape
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Define convolution with x and W
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Define max pooling with x and W
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# 1st ConvNet layer
## 32 Kernel Filter with size 5x5 on 1 input channel
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 1st Pooling layer
## 2x2 max pooling
h_pool1 = max_pool_2x2(h_conv1)

# 2nd ConvNet layer
## 64 Kernel Filter with size 5x5 on 32 input channel
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 2nd Pooling layer
## 2x2 max pooling
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Apply dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 텐서보드로 그래프 출력
with tf.Session() as sess:
  writer = tf.summary.FileWriter('/tmp/tf-slim-tutorial', sess.graph)
  sess.run(tf.global_variables_initializer())
  writer.close()