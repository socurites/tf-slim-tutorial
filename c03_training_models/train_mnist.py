# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
손실함수를 정의하는 방법
MNIST examples
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import c01_defining_models.s04_examples.mnist_deep_step_by_step_slim as mnist_model
from utils.dataset_utils import load_batch
from datasets import tf_record_dataset

tf.logging.set_verbosity(tf.logging.INFO)

'''
# 훈련 데이터 로드
'''
mnist_tfrecord_dataset = tf_record_dataset.TFRecordDataset(
    tfrecord_dir='/home/itrocks/Git/Tensorflow/tf-slim-tutorial/raw_data/mnist/tfrecord',
    dataset_name='mnist',
    num_classes=10)
# Selects the 'train' dataset.
dataset = mnist_tfrecord_dataset.get_split(split_name='train')
images, labels, _ = load_batch(dataset)

'''
# 모델 정의
'''
logits = mnist_model.mnist_convnet(images)


'''
# 손실함수 정의
'''
loss = slim.losses.softmax_cross_entropy(logits, labels)
total_loss = slim.losses.get_total_loss()

"""
반복 훈련하는 방법
MNIST examples
#
# 옵티마이저 선택하기
# 모델 체크포인트 저장하기
"""

'''
# 옵티마이저 정의: Adam
'''
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)


'''
# 메트릭 정의
'''
predictions = tf.argmax(logits, 1)
targets = tf.argmax(labels, 1)

correct_prediction = tf.equal(predictions, targets)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('losses/Total', total_loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()


'''
# 훈련하기
'''
# logging 경로 설정
log_dir = '/tmp/tfslim_model/'
if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)

# 훈련 오퍼레이션 정의
train_op = slim.learning.create_train_op(total_loss, optimizer)

final_loss = slim.learning.train(
    train_op,
    log_dir,
    number_of_steps=2000,
    summary_op=summary_op,
    save_summaries_secs=30,
    save_interval_secs=30)

print('Finished training. Final batch loss %f' % final_loss)
