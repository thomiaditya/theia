# This module contains the model class and its methods for training.
import os
import tensorflow as tf
import numpy as np
import config

class Model(tf.keras.Model):
    def __init__(self):
        """
        Initialize the model class.
        """
        super(Model, self).__init__()

        self.config = config.model_config

    def call(self, inputs):
        """
        This method is called when the model is called.
        """
        return self.model(inputs)

    def model(self, inputs):
        """
        This method is called when the model is called.
        """
        # This is the model definition.
        inputs = tf.keras.layers.Input(shape=self.config["input_shape"])
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        x = tf.keras.layers.Dense(units=10, activation="softmax")(x)
        return x

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
                    # Forward pass.
                    predictions = self(images)
                    # Compute the loss.
                    loss = self.config["loss"](labels, predictions)

                # Compute the gradients.
                gradients = tape.gradient(loss, self.trainable_variables)
                # Update the weights.
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
                # Log the training loss.
                if batch % 200 == 0:
                    print("Seen so far: {} samples".format(batch * self.config["batch_size"]))
                    print("Train Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch, loss.numpy()))

            # Iterate over the validation data.
            for batch, (images, labels) in enumerate(val_dataset):
                # Compute the loss.
                predictions = self(images)
                loss = self.config["loss"](labels, predictions)
                # Print the loss.
                print("Val Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch, loss.numpy()))

    def evaluate(self, test_dataset):
        """
        This method is called when the model is evaluated.
        """
        # This is the evaluation loop.
        for batch, (images, labels) in enumerate(test_dataset):
            # Compute the loss.
            predictions = self(images)
            loss = self.config["loss"](labels, predictions)
            # Print the loss.
            print("Batch: {}, Loss: {}".format(batch, loss.numpy()))