# This file will contains the custom callback for the model.
import os
import tensorflow as tf
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("Epoch: {}".format(epoch))