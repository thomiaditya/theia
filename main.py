# This module contains the main function which is called when the program is run.
import os
import tensorflow as tf
import numpy as np
from theia.model.model import Model
import tensorflow_datasets as tfds

model = Model()

# Load the data
(train_data, test_data), info = tfds.load("mnist", split=["train", "test"], as_supervised=True, with_info=True)

# Preprocess the data into 0 to 1
train_data = train_data.map(lambda image, label: (tf.cast(image, dtype="float32") / 255.0, tf.one_hot(label, 10)))
test_data = test_data.map(lambda image, label: (tf.cast(image, dtype="float32") / 255.0, tf.one_hot(label, 10)))

# Batch the data to adding a new Dimension to the data (None, 28, 28, 1)
train_data = train_data.batch(32)
test_data = test_data.batch(32)

# model.train(train_data, test_data)

model.load()

print(model.predict(test_data))

model.save()