# Model class for the Retrieval Recommender System.

import os
import sys
import numpy as np
from typing import Dict, List, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from ..config.recommender import hyperparameters as hp, dataset as ds
from ..config.recommender.retrieval import definition
from alive_progress import alive_bar
import wandb


class RetrievalModel(tf.keras.Model):
    """
    Model class for the Retrieval Recommender System. Retrieval Model will retrieve the most related items to the input item.
    """

    def __init__(self):
        """
        Initialize the Retrieval Model.
        """
        super().__init__()
        self.hyperparameters = hp.hyperparameters

        # Get the unique vocabs from the dataset.
        self.unique_queries, self.unique_candidates = ds.get_unique_vocabs()

        # Build the model.
        self.query_tower, self.candidate_tower = definition.build_model(
            self.unique_queries, self.unique_candidates)

        # Build the metrics. The metrics will be added in self.factorized_top_k.
        self.build_metrics()

        # Instantiate Wandb if use_wandb is set to True.
        self.instantiate_wandb()

        # Build the loss function. The loss function will be added in hyperparameters.
        self.build_loss_function()

    def build_metrics(self):
        """
        Build the metrics for the Retrieval Model.
        """
        try:
            self.factorized_top_k = tfrs.metrics.FactorizedTopK(
                candidates=ds.get_candidates().batch(128).map(self.candidate_tower))
            return 0
        except Exception as e:
            print(e)
            sys.exit(1)

    def build_loss_function(self):
        """
        Build the loss function for the Retrieval Model.
        """
        try:
            self.hyperparameters["loss"] = tfrs.tasks.Retrieval(
                metrics=self.factorized_top_k)
            return 0
        except Exception as e:
            print(e)
            sys.exit(1)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
        """
        Compute the loss for the Retrieval Model. We import from user defined loss in dataset.py.
        """
        return ds.compute_loss(self, features, training)

    def metrics_to_string(self, metrics):
        """
        Convert the metrics to a string.
        """
        metrics_str = ""
        for key, value in metrics.items():
            if type(value) == np.ndarray:
                value = ", ".join(["{:.3f}".format(v)
                                  for v in value])
                metrics_str += "{}: {} | ".format(key, value)
                continue

            metrics_str += "{}: {:.3f} | ".format(key, value)
        return metrics_str

    def instantiate_wandb(self):
        """
        Instantiate Wandb.
        """
        if self.hyperparameters["use_wandb"]:
            wandb.init(project=self.hyperparameters["name"],
                       name=self.hyperparameters["id"],
                       config=self.hyperparameters)

    def fit(self):
        """
        Fit the Retrieval Model.
        """

        # Prompt the user that the model is training using yellow color.
        print("\033[33mTraining the Retrieval Model...\033[0m")

        # Compile the model.
        self.compile(optimizer=self.hyperparameters["optimizer"])

        # Cache the dataset.
        cached_train = ds.get_train_dataset()

        # Get the number of batches.
        num_batches = len(cached_train)

        with alive_bar(num_batches * self.hyperparameters["epochs"], ctrl_c=False, manual=False, dual_line=True, spinner="pulse") as bar:

            # Implement custom training loop.
            for epoch in range(self.hyperparameters["epochs"]):

                # Set bar title to current epoch.
                bar.title = "Epoch {}/{}".format(epoch + 1,
                                                 self.hyperparameters["epochs"])

                # Train the model.
                for step, features in enumerate(cached_train):

                    # Get the metrics from the training step.
                    metrics = self.train_step(features)

                    # Update the bar.
                    if step % self.hyperparameters["log_every"] == 0 or step == 0:
                        metrics_str = self.metrics_to_string(metrics)

                    # Text for the bar.
                    bar.text = metrics_str

                    # Update the bar.
                    bar()

                # Print the final metrics if not using Wandb.
                if not self.hyperparameters["use_wandb"]:
                    print("epoch {}: {}".format(
                        epoch + 1, self.metrics_to_string(metrics)))
                else:
                    wandb.log({"epoch": epoch + 1, **metrics})

    def train_step(self, features):
        """
        Train the model for one step.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(
                features, not self.hyperparameters["metrics_calculation"])

            # Regularization.
            regularization_loss = sum(self.losses)

            # Compute the total loss.
            total_loss = loss + regularization_loss

        # Compute the gradients.
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update the weights.
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        # Update the metrics.
        metrics = {}
        if self.hyperparameters["metrics_calculation"]:
            metrics = {"factorized_top_k": np.array([v.numpy()
                                                     for v in self.factorized_top_k.result()])}

        # Add loss to the metrics.
        metrics["loss"] = loss
        metrics["total_loss"] = total_loss

        # Log the metrics.
        return metrics

    def eval_step(self, features):
        """
        Evaluate the model for one step.
        """
        loss = self.compute_loss(features, False)

        # Regularization.
        regularization_loss = sum(self.losses)

        # Compute the total loss.
        total_loss = loss + regularization_loss

        # Update the metrics.
        metrics = {"factorized_top_k": np.array([v.numpy()
                                                 for v in self.factorized_top_k.result()])}

        # Add loss to the metrics.
        metrics["loss"] = loss
        metrics["total_loss"] = total_loss

        # Log the metrics.
        return metrics

    def evaluate(self):
        """
        Evaluate the model.
        """

        # Prompt to user that evaluation is starting using yellow.
        print("\033[33mEvaluation starting...\033[0m")

        # Cache the dataset.
        cached_eval = ds.get_test_dataset()

        # Get the number of batches.
        num_batches = len(cached_eval)

        with alive_bar(num_batches, ctrl_c=False, manual=False, dual_line=True, spinner="pulse") as bar:

            # Set bar title to current epoch.
            bar.title = "Evaluating: "

            # Train the model.
            for step, features in enumerate(cached_eval):

                # Get the metrics from the training step.
                metrics = self.eval_step(features)

                # Update the bar.
                if step % self.hyperparameters["log_every"] == 0 or step == 0:
                    metrics_str = self.metrics_to_string(metrics)

                # Text for the bar.
                bar.text = metrics_str

                # Update the bar.
                bar()

        # Print the metrics.
        print("evaluation: {}".format(self.metrics_to_string(metrics)))

    def predict(self, index):
        """
        Predict the model.
        """
        index = tfrs.layers.factorized_top_k.BruteForce(self.query_tower)

        index.index_from_dataset(ds.get_candidates().batch(100).map(
            lambda title: (title, self.candidate_tower(title))))

        _, titles = index(np.array([index]))

        return titles
