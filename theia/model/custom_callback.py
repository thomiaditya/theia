# This file will contains the custom callback for the model.
import os
import tensorflow as tf
import numpy as np
import wandb

class CustomCallback(tf.keras.callbacks.Callback):
  """
  Note that on_epoch_(end|start) and on_train_(end|start) is called after each epoch and its passing additional logs which are "wandb" (if you enable use_wandb) and "validation_data" keys.
  """
  def on_epoch_end(self, epoch, logs=None):
    images = logs["validation_data"].take(1)

    # Iterate through the data and send to wandb for visualization
    images_sent = []
    for image, label in images:
      for i, img in enumerate(image):
        images_sent.append(wandb.Image(img.numpy(), caption="Label: {}, Prediction: {}".format(np.argmax(label[i]), np.argmax(self.model.predict(np.expand_dims(img.numpy(), axis=0))))))

    # Send the images to wandb
    logs["wandb"].log({
      "image_predictions": images_sent
    })

    print("Image predictions sent to wandb")