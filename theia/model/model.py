# This module contains the model class and its methods for training.
from contextlib import redirect_stdout
import os
import tensorflow as tf
import numpy as np
from model import config
import progressbar as pb

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
            print("Epoch ", epoch + 1)

            # Use progressbar to show the progress of the training.
            widgets = [pb.Percentage(), " ", pb.Bar(marker="="), " ", pb.ETA()]
            pbar = pb.ProgressBar(widgets=widgets, maxval=len(train_dataset)).start()

            # Iterate over the training data.
            for batch, (images, labels) in enumerate(train_dataset):
                
                # Open Gradient Tape.
                with tf.GradientTape() as tape:
                    # Compute the loss.
                    predictions = self.model(images, training=True)
                    loss = self.config["loss"](labels, predictions)

                # Calculate the accuracy.
                self.config["metrics"][0](labels, predictions)
                
                # Compute the gradients.
                gradients = tape.gradient(loss, self.model.trainable_weights)
                
                # Apply the gradients.
                self.config["optimizer"].apply_gradients(zip(gradients, self.model.trainable_weights))

                # Update the progressbar.
                pbar.update(batch)
                    
            # Iterate over the validation data.
            for batch, (images, labels) in enumerate(val_dataset):
                # Compute the loss.
                predictions = self.model(images, training=False)
                loss = self.config["loss"](labels, predictions)