# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
Dataset 클래스

# Ref: TensorFlow-Slim image classification library
       https://github.com/tensorflow/models/tree/master/slim
#
# 아래 코드는 flowers Dataset 클래스를 정의하는 코드임
# 아래 코드를 약간 변형해서 사용
# https://github.com/tensorflow/models/blob/master/slim/datasets/flowers.py
"""

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets import dataset_utils

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer',
}


class TFRecordDataset:
    def __init__(self, tfrecord_dir, dataset_name, num_classes):
        self.tfrecord_dir = tfrecord_dir
        self.dataset_name = dataset_name
        self.num_classes = num_classes

    def __get_num_samples__(self, split_name):
        # Count the total number of examples in all of these shard
        num_samples = 0
        file_pattern_for_counting = self.dataset_name + '_' + split_name
        tfrecords_to_count = [os.path.join(self.tfrecord_dir, file) for file in os.listdir(self.tfrecord_dir) if
                              file.startswith(file_pattern_for_counting)]
        for tfrecord_file in tfrecords_to_count:
            for record in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        return num_samples

    def get_split(self, split_name):
        splits_to_sizes = self.__get_num_samples__(split_name)

        if split_name not in ['train', 'validation']:
            raise ValueError('split name %s was not recognized.' % split_name)

        file_pattern = self.dataset_name + '_' + split_name + '_*.tfrecord'
        file_pattern = os.path.join(self.tfrecord_dir, file_pattern)
        reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        if dataset_utils.has_labels(self.tfrecord_dir):
            labels_to_names = dataset_utils.read_label_file(self.tfrecord_dir)

        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=splits_to_sizes,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=self.num_classes,
            labels_to_names=labels_to_names)
