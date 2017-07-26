# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
TFRecord Dataset 로드하기
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets import tf_record_dataset

# create MNIST dataset
mnist_tfrecord_dataset = tf_record_dataset.TFRecordDataset(tfrecord_dir='/home/itrocks/Git/Tensorflow/tf-slim-tutorial/raw_data/mnist/tfrecord',
                                                  dataset_name='mnist',
                                                  num_classes=10)
# Selects the 'train' dataset.
dataset = mnist_tfrecord_dataset.get_split(split_name='train')

# Creates a TF-Slim DataProvider which reads the dataset in the background during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

import matplotlib.pyplot as plt
with tf.Session() as sess:
  with slim.queues.QueueRunners(sess):
    for i in range(4):
      np_image, np_label = sess.run([image, label])
      height, width, _ = np_image.shape
      class_name = name = dataset.labels_to_names[np_label]

      plt.figure()
      plt.imshow(np_image)
      plt.title('%s, %d x %d' % (name, height, width))
      plt.axis('off')
      plt.show()