# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
레이어를 정의하는 방법
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import misc

"""
nativeTF에서 컨볼루션 레이어를 정의하는 방법
# 1. 가중치와 바이어스에 대한 변수를 생성한다
# 2. 이전 레이어의 출력을 입력으로 사용하여, 가중치에 대해 컨볼루션 오퍼레이션을 정의한다.
# 3. 위의 출력에 바이어스를 더한다
# 4. 활성화 함수를 적용한다
"""

input_val = tf.placeholder(tf.float32, [16, 32, 32, 64])

with tf.name_scope('conv1_1') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1, name='weight'))
  conv = tf.nn.conv2d(input=input_val, filter=kernel, strides=[1,1,1,1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias, name=scope)


"""
TF-Slim에서 컨볼루션 레이어를 정의하는 방법
"""

# padding='SAME' is default
# strindes=[1,1,1,1] is default
net = slim.conv2d(inputs=input_val, num_outputs=128, kernel_size=[3,3], scope='conv1_1')


"""
메타 오퍼레이션: repeat
"""
# VGG network 일부
net1 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.name_scope('test1') as scope:
  net1 = slim.conv2d(net1, 256, [3,3], scope='conv3_1')
  net1 = slim.conv2d(net1, 256, [3,3], scope='conv3_2')
  net1 = slim.conv2d(net1, 256, [3,3], scope='conv3_3')
  net1 = slim.max_pool2d(net1, [2,2], scope='pool2')

# for loop 사용
net2 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.name_scope('test2') as scope:
  for i in range(3):
    net2 = slim.conv2d(net2, 256, [3,3], scope='conv3_%d' % (i+1))
  net2 = slim.max_pool2d(net2, [2,2], scope='pool2')

# TF-Slim repeat 사용
net3 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.name_scope('test3') as scope:
  net3 = slim.repeat(net3, 3, slim.conv2d, 256, [3,3], scope='conv3')
  net3 = slim.max_pool2d(net2, [2,2], scope='pool2')


"""
메타 오퍼레이션: stack
"""
# MLP 일부
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 4])
  mlp1 = slim.fully_connected(inputs=input_val, num_outputs=32, scope='fc/fc_1')
  mlp1 = slim.fully_connected(inputs=mlp1, num_outputs=64, scope='fc/fc_2')
  mlp1 = slim.fully_connected(inputs=mlp1, num_outputs=128, scope='fc/fc_3')

print([node.name for node in g.as_graph_def().node])


# TF-Slim stack 사용
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 4])
  mlp2 = slim.stack(input_val, slim.fully_connected, [32, 64, 128], scope='fc')

  print([node.name for node in g.as_graph_def().node])

  train_writer = tf.summary.FileWriter('/tmp/tf-slim-tutorial', g)

  train_writer.close()


# ConvNet 일부
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 32, 32, 8])
  conv1 = slim.conv2d(input_val, 32, [3,3], scope='core/core_1')
  conv1 = slim.conv2d(conv1, 32, [1, 1], scope='core/core_2')
  conv1 = slim.conv2d(conv1, 64, [3, 3], scope='core/core_3')
  conv1 = slim.conv2d(conv1, 64, [1, 1], scope='core/core_4')

  train_writer = tf.summary.FileWriter('/tmp/tf-slim-tutorial', g)

  train_writer.close()


# TF-Slim stack 사용
g = tf.Graph()
with g.as_default():
  input_val = tf.placeholder(tf.float32, [16, 32, 32, 8])
  conv2 = slim.stack(input_val, slim.conv2d, [(32,[3,3]), (32,[1,1]), (64,[3,3]), (64,[1,1])], scope='core')