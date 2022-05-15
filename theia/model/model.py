# This module contains the model class and its methods for training.
import os
import tensorflow as tf
import numpy as np
from model import config

class Model():
    def __init__(self):
        """
        Initialize the model class.
        """
        super(Model, self).__init__()

        self.config = config.config
        self.model = config.model_definition

    def train(self, train_dataset, val_dataset):
        """
        This method is called when the model is trained.
        """
        # This is the training loop.
        for epoch in range(self.config["epochs"]):
            
            # Iterate over the training data.
            for batch, (images, labels) in enumerate(train_dataset):
                # Open Gradient Tape.
                with tf.GradientTape() as tape:
                    # Compute the loss.
                    predictions = self.model(images, training=True)
                    loss = self.config["loss"](labels, predictions)
                
                # Compute the gradients.
                gradients = tape.gradient(loss, self.model.trainable_weights)
                # Apply the gradients.
                self.config["optimizer"].apply_gradients(zip(gradients, self.model.trainable_weights))
                # Log the training loss.
                print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch, loss))

            # Iterate over the validation data.
            for batch, (images, labels) in enumerate(val_dataset):
                # Compute the loss.
                predictions = self.model(images, training=False)
                loss = self.config["loss"](labels, predictions)
                # Log the validation loss.
                if batch % 200 == 0:
                    print("Validation Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch, loss.numpy()))