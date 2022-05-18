# This module contains the main function which is called when the program is run.
import os
import tensorflow as tf
import numpy as np
from theia.model import Model
import tensorflow_datasets as tfds
import wandb

model = Model()

# Load the data
(train_data, test_data), info = tfds.load("mnist", split=["train", "test"], as_supervised=True, with_info=True)

# Preprocess the data into 0 to 1
train_data = train_data.map(lambda image, label: (tf.cast(image, dtype="float32") / 255.0, tf.one_hot(label, 10)))
test_data = test_data.map(lambda image, label: (tf.cast(image, dtype="float32") / 255.0, tf.one_hot(label, 10)))

# Batch the data to adding a new Dimension to the data (None, 28, 28, 1)
train_data = train_data.batch(32)
test_data = test_data.batch(32)

model.train(train_data, test_data)

# model.load()

# images = test_data.take(1)

# # Iterate through the data and send to wandb for visualization
# images_sent = []
# for image, label in images:
#   for i, img in enumerate(image):
#     images_sent.append(wandb.Image(img.numpy(), caption="Label: {}, Prediction: {}".format(np.argmax(label[i]), np.argmax(model.predict(np.expand_dims(img.numpy(), axis=0))))))

# print(len(images_sent))

# model.wandb_log({
#   "image_predictions": images_sent
# })

model.save()