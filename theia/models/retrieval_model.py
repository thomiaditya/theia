import os
import sys
from typing import Dict
import tensorflow as tf
import numpy as np
from ..config.recommender import retrieval_definition as rd, params, dataset as ds
from ..utils import Logger
from alive_progress import alive_bar


class RetrievalModel():
    def __init__(self):
        """
        Definition for the model definition in Retrieval Definition Class.
        """
        # Load the model definition from the config file.
        self.model = rd.RetrievalDefinition()
        self.logger = Logger()

        # Prompt the user that the model is created.
        self.logger.write(
            "Model is created successfully", level="INFO")

    def train(self):
        """
        Training the dataset from dataset file using alive progress bar.
        """
        # Prompt the user that the training is starting.
        self.logger.write(
            "Training is starting...", "WARNING")

        # Get the train data.
        train_data = ds.get_train_data()
        train_data = train_data.batch(params.train_batch_size)

        # Get the number of batches.
        num_batches = len(train_data)

        # Start alive progress bar.
        with alive_bar(num_batches * params.epochs, ctrl_c=False, manual=False, dual_line=True, spinner="pulse") as bar:

            # Iterate over the epochs.
            for epoch in range(params.epochs):

                # Set alive progress bar title to epoch.
                bar.title = "Epoch {}/{}".format(epoch + 1, params.epochs)

                # Iterate over the batches.
                for batch, features in enumerate(train_data):

                    # Compute the loss.
                    metrics = self.train_step(features)

                    # Metrics to string.
                    metrics_string = self.metrics_to_string(metrics)

                    # Update the alive progress bar text.
                    bar.text = metrics_string

                    # Update the progress bar.
                    bar()

                # Print the result metrics and epoch.
                print("epoch {}: {}".format(epoch + 1, metrics_string))

        # Prompt the user that the training is finished.
        self.logger.write(
            "Training is finished.", "WARNING")

    def metrics_to_string(self, metrics) -> str:
        """
        Convert metrics to string.
        """
        _metrics_string = ""
        for key, value in metrics.items():
            _type = type(value)
            if _type == np.ndarray:
                _metrics_string += "{}: ".format(key)
                _metrics_string += ", ".join(
                    ["{:.4f}".format(value) for value in value])
                continue

            _metrics_string += "{}: {:.4f} | ".format(key, value)

        return _metrics_string

    def train_step(self, features):
        """
        Training step for the model.
        """
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:

            # Loss computation.
            loss = self.model.compute_loss(features)

            # Handle regularization losses as well.
            regularization_loss = sum(self.model.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        params.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        metrics = {}

        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        metrics["factorized_top_k"] = np.array(
            [metric.result() for metric in self.model.metrics])

        return metrics

    def recommend(self, index):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
