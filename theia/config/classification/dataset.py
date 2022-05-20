# This module is for loading the dataset from tfds and preparing it for training.

import tensorflow_datasets as tfds
import tensorflow as tf
from .model_config import model_config as config


def load_datasets():
    """
    This function loads the dataset from tfds and prepares it for training.

    return train_dataset, val_dataset, info
    """

    # Load the dataset from tfds
    (train_dataset, val_dataset), info = tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    # Prepare the dataset for training
    train_dataset = train_dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0,
                      tf.one_hot(y, depth=config["output_shape"])))
    val_dataset = val_dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0,
                      tf.one_hot(y, depth=config["output_shape"])))

    # Batch the datasets
    train_dataset = train_dataset.batch(config["batch_size"])
    val_dataset = val_dataset.batch(config["batch_size"])

    return train_dataset, val_dataset, info
