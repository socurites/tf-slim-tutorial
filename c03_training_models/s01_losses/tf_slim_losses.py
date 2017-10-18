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

# Load the images and labels.
from datasets import tf_record_dataset

# create MNIST dataset
mnist_tfrecord_dataset = tf_record_dataset.TFRecordDataset(
    tfrecord_dir='/home/itrocks/Git/Tensorflow/tf-slim-tutorial/raw_data/mnist/tfrecord',
    dataset_name='mnist',
    num_classes=10)
# Selects the 'train' dataset.
dataset = mnist_tfrecord_dataset.get_split(split_name='train')
images, labels, _ = load_batch(dataset)

# 모델 생성
logits = mnist_model.mnist_convnet(images)

print(logits)
print(labels)

# 손실 함수 정의
loss = slim.losses.softmax_cross_entropy(logits, labels)

# multi-taks model의 경우 multi loss 추가한 후 total_loss 활용
total_loss = slim.losses.get_total_loss()

"""
반복 훈련하는 방법
MNIST examples
#
# 옵티마이저 선택하기
# 모델 체크포인트 저장하기
"""

log_dir = '/tmp/tfslim_model/'
if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)

tf.logging.set_verbosity(tf.logging.INFO)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train_op = slim.learning.create_train_op(total_loss, optimizer)

predictions = tf.argmax(logits, 1)
targets = tf.argmax(labels, 1)

correct_prediction = tf.equal(predictions, targets)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('losses/Total', total_loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

final_loss = slim.learning.train(
    train_op,
    log_dir,
    number_of_steps=1000,
    summary_op=summary_op,
    save_summaries_secs=30,
    save_interval_secs=60)

print('Finished training. Fianl batch lose %f' % final_loss)
