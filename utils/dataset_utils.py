import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_batch(dataset, batch_size=32, height=28, width=28, num_classes=10, is_training=True):
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
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image = tf.to_float(image)

    # label: one-hot encoding
    one_hot_labels = slim.one_hot_encoding(label, num_classes)

    # Batch it up.
    images, labels = tf.train.batch(
        [image, one_hot_labels],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, labels, dataset.num_samples
