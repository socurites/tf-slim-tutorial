# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
TF-Slim에서 추가된 slim.arg_scope()
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
아래의 코드는 하이퍼파라미터/초기화 등 중복이 있이며, 가독성도 떨어짐
"""
with tf.variable_scope('test1'):
    input_val = tf.placeholder(tf.float32, [16, 300, 300, 64])

    net1 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4, padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
    net1 = slim.conv2d(inputs=net1, num_outputs=128, kernel_size=[11, 11], padding='VALID',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
    net1 = slim.conv2d(inputs=net1, num_outputs=256, kernel_size=[11, 11], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')

"""
1차 리팩토링
"""
with tf.variable_scope('test2'):
    padding = 'SAME'
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    regularizer = slim.l2_regularizer(0.0005)
    net2 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4,
                      padding=padding,
                      weights_initializer=initializer,
                      weights_regularizer=regularizer,
                      scope='conv1')
    net2 = slim.conv2d(inputs=net2, num_outputs=128, kernel_size=[11, 11],
                      padding='VALID',
                      weights_initializer=initializer,
                      weights_regularizer=regularizer,
                      scope='conv2')
    net2 = slim.conv2d(inputs=net2, num_outputs=256, kernel_size=[11, 11],
                      padding=padding,
                      weights_initializer=initializer,
                      weights_regularizer=regularizer,
                      scope='conv3')


"""
2차 리팩토링: slim.arg_scope() 사용
# 공통된 인자는 arg_scope()에 정의.
# 다른 인자만 재정의
"""
with tf.variable_scope('test3'):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net3 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4, scope='conv1')
        net3 = slim.conv2d(inputs=net3, num_outputs=128, kernel_size=[11, 11], padding='VALID', scope='conv2')
        net3 = slim.conv2d(inputs=net3, num_outputs=256, kernel_size=[11, 11], scope='conv3')


"""
3차 리팩토링: slim.arg_scope() 중첩 가능
"""
with tf.variable_scope('test5'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
      with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        net4 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4, scope='conv1')
        net4 = slim.conv2d(inputs=net4, num_outputs=256, kernel_size=[5, 5],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                          scope='conv2')
        net4 = slim.fully_connected(inputs=net4, num_outputs=1000, activation_fn=None, scope='fc')
