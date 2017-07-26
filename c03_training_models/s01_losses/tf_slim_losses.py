# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
손실함수를 정의하는 방법
MNIST examples
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import c01_defining_models.s04_examples.mnist_deep_step_by_step_slim as mnist_model

def load_batch(dataset, batch_size=32, height=28, width=28, is_training=True):
  """Loads a single batch of data.

  Args:
    dataset: The dataset to load.
    batch_size: The number of images in the batch.
    height: The size of each image after preprocessing.
    width: The size of each image after preprocessing.
    is_training: Whether or not we're currently training or evaluating.

  Returns:
    images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
    images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
    labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
  """
  # Creates a TF-Slim DataProvider which reads the dataset in the background during both training and testing.
  provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
  [image, label] = provider.get(['image', 'label'])

  # image: resize with crop
  image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
  image = tf.to_float(image)

  # label: one-hot encoding
  one_hot_labels = slim.one_hot_encoding(label, 10)

  # Batch it up.
  images, labels = tf.train.batch(
    [image, one_hot_labels],
    batch_size=batch_size,
    num_threads=1,
    capacity=2 * batch_size)

  return images, labels



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

# 손실 함수 정의
loss = slim.losses.softmax_cross_entropy(predictions, labels)

tf.logging.set_verbosity(tf.logging.INFO)

total_loss = slim.losses.get_total_loss()

# Create some summaries to visualize the training process:
tf.summary.scalar('losses/Total Loss', total_loss)

# Specify the optimizer and create the train op:
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = slim.learning.create_train_op(total_loss, optimizer)

train_dir = '/tmp/tfslim_model/'
# Run the training:
final_loss = slim.learning.train(
  train_op,
  logdir=train_dir,
  number_of_steps=100,  # For speed, we just do 1 epoch
  save_summaries_secs=1)

print('Finished training. Final batch loss %d' % final_loss)


#
# # Evaluation
# dataset = mnist_tfrecord_dataset.get_split(split_name='validation')
# images, labels = load_batch(dataset)
#
#
# predictions = mnist_convnet(images)
# predictions = tf.argmax(predictions, 1)
#
# # Define the metrics:
# names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#   'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
#   'eval/Recall@5': slim.metrics.streaming_recall_at_k(predictions, labels, 5),
# })
#
# print('Running evaluation Loop...')
# checkpoint_path = tf.train.latest_checkpoint(train_dir)
# metric_values = slim.evaluation.evaluate_once(
#   master='',
#   checkpoint_path=checkpoint_path,
#   logdir=train_dir,
#   eval_op=names_to_updates.values(),
#   final_op=names_to_values.values())
#
# names_to_values = dict(zip(names_to_values.keys(), metric_values))
# for name in names_to_values:
#   print('%s: %f' % (name, names_to_values[name]))