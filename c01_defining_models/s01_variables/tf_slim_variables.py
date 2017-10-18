# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
TF-Slim에서 variable을 사용하는 방법
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
# nativeTF의 아래 코드와 TF-Slim의 아래 코드는 동일
with tf.device("/cpu:0"):
  weight_4 = tf.Variable(tf.truncated_normal(shape=[784, 200], mean=1.5, stddev=0.35), name="w4")
"""

weight_4 = slim.variable('w4',
                         shape=[784, 200],
                         initializer=tf.truncated_normal_initializer(mean=1.5, stddev=0.35),
                         device='/CPU:0')

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    val_w4 = sess.run(weight_4)

"""
모델 변수(model variables) vs. 일반 변수(regular variable)
"""

# 모델 변수 생성하기
weight_5 = slim.model_variable('w5',
                               shape=[10, 10, 3, 3],
                               initializer=tf.truncated_normal_initializer(stddev=0.1),
                               regularizer=slim.l2_regularizer(0.05),
                               device='/CPU:0')

model_variables = slim.get_model_variables()

print([var.name for var in model_variables])

# 일반 변수 생성하기
my_var_1 = slim.variable('mv1',
                         shape=[20, 1],
                         initializer=tf.zeros_initializer())

model_variables = slim.get_model_variables()
all_variables = slim.get_variables()

print([var.name for var in model_variables])
print([var.name for var in all_variables])
