# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
native TF에서 variable을 사용하는 방법
# Creation
# Initialization
# Saving
# Loading / Restoring
#
# Ref: TensorFlow > Prgorammer's Guide > Variables: Creation, Initialization, Saving, and Loading
#      https://www.tensorflow.org/programmers_guide/variables
"""

import tensorflow as tf
import matplotlib.pyplot as plt

"""
Creation
# 미리 정해진 상수(constant) 또는
# - tf.zeros
# - tf.ones
# 초기화 메커니즘에 따라 변수를 생성
# - 시퀀스 텐서
#    - tf.linspace
#    - tf.range
# - 랜덤 텐서
#    - tf.random_normal
#    - tf.truncated_normal
# Ref: TensorFlow > API > Constants, Sequences, and Random Values
#      https://www.tensorflow.org/api_guides/python/constant_op
"""
# 다양한 방법으로 변수 생성
bias_1 = tf.Variable(tf.zeros(shape=[200]), name="b1")
weight_1 = tf.Variable(tf.lin_space(start=0.0, stop=12.0, num=3), name="w1")
weight_2 = tf.Variable(tf.range(start=0.0, limit=12.0, delta=3), name="w2")
weight_3 = tf.Variable(tf.random_normal(shape=[784, 200], mean=1.5, stddev=0.35), name="w3")
weight_4 = tf.Variable(tf.truncated_normal(shape=[784, 200], mean=1.5, stddev=0.35), name="w4")

print(weight_1)

"""
Initialization
# 다른 오퍼레이션을 실행하기 전에 변수 초기화를 선행
# tf.global_variables_initializer()를 이용하여 모든 변수 초기화를 실행
"""
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    val_b1 = sess.run(bias_1)
    val_w1, val_w2, val_w3, val_w4 = sess.run([weight_1, weight_2, weight_3, weight_4])

    print(type(val_b1))
    print(val_w1.shape)

    # 그래프로 변수 확인하기
    plt.subplot(221)
    plt.hist(val_w1)
    plt.title('val_w1_linspace')
    plt.grid(True)

    plt.subplot(222)
    plt.hist(val_w2)
    plt.title('val_w2_range')
    plt.grid(True)

    plt.subplot(223)
    plt.hist(val_w3)
    plt.title('val_w3_random_normal')
    plt.grid(True)

    plt.subplot(224)
    plt.hist(val_w4)
    plt.title('val_w2_truncated_normal')
    plt.grid(True)

    plt.show()

# Device placement
# 변수를 특정 디바이스에 할당
with tf.device("/cpu:0"):
    bias_2 = tf.Variable(tf.ones(shape=[200]), name="b2")

print(bias_1)
print(bias_2)

"""
Saving / Restoring
# tf.train.Saver 객체를 이용하여 변수를 체크포인트 파일로 저장/로드 가능
"""
model_path = "/tmp/tx-01.ckpt"

# 저장
bias_3 = tf.add(bias_1, bias_2, name='b3')
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)

    val_b3 = sess.run(bias_3)
    print(val_b3)

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

# 로드
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_path)
    print("Model restored")

    # access tensor by name directly
    val_b3 = sess.run('b3:0')
    print(val_b3)

    # get tensor by name
    graph = tf.get_default_graph()
    b3 = graph.get_tensor_by_name("b3:0")
    val_b3 = sess.run(b3)
    print(val_b3)
