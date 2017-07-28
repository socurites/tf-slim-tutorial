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
images, labels = load_batch(dataset)

# 모델 생성
predictions = mnist_model.mnist_convnet(images)

print(predictions)
print(labels)

# 손실 함수 정의
loss = slim.losses.softmax_cross_entropy(predictions, labels)

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

final_loss = slim.learning.train(
  train_op,
  log_dir,
  number_of_steps=5000,
  save_summaries_secs=300,
  save_interval_secs=600)

print('Finished training. Fianl batch lose %f' %final_loss)
