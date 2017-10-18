# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
TFRecord Dataset 로드하기
# 1. TFRecord 포맷 데이터을 읽어서 변환할 수 있도록 slim.dataset.Dataset 클래스를 정의한다.
# 2. 데이터를 피드하기 위한 slim.dataset_data_provider.DatasetDataProvider를 생성한다.
# 3. 네트워크 모델의 입력에 맞게 전처리 작업 및 편의를 위한 one-hot 인코딩 작업을 한 후, tf.train.batch를 생성한다.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets import tf_record_dataset
from utils.dataset_utils import load_batch

"""
# slim.dataset.Dataset 클래스를 정의
"""
TF_RECORD_DIR = '/home/itrocks/Git/Tensorflow/tf-slim-tutorial/raw_data/mnist/tfrecord'
mnist_tfrecord_dataset = tf_record_dataset.TFRecordDataset(tfrecord_dir=TF_RECORD_DIR,
                                                           dataset_name='mnist',
                                                           num_classes=10)
# train 데이터셋 생성
dataset = mnist_tfrecord_dataset.get_split(split_name='train')

"""
# slim.dataset_data_provider.DatasetDataProvider를 생성
"""
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

# 테스트
import matplotlib.pyplot as plt
with tf.Session() as sess:
    with slim.queues.QueueRunners(sess):
        plt.figure()
        for i in range(4):
            np_image, np_label = sess.run([image, label])
            height, width, _ = np_image.shape
            class_name = name = dataset.labels_to_names[np_label]

            plt.subplot(2, 2, i+1)
            plt.imshow(np_image)
            plt.title('%s, %d x %d' % (name, height, width))
            plt.axis('off')
        plt.show()


'''
tf.train.batch를 생성
'''
images, labels, _ = load_batch(dataset)