# Model class for the Retrieval Recommender System.

import os
import sys
import tempfile
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

        # Set the training flag.
        self.is_training = self.hyperparameters["mode"] == "training"

        # Save the id.
        self.id = self.hyperparameters["id"]

        if not self.is_training:
            return

        # Get the unique vocabs from the dataset.
        self.unique_queries, self.unique_candidates = ds.get_unique_vocabs()

        # Add checkpoint manager.
        self.add_checkpoint_manager()

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

    def decorator(function):
        """
        Decorator for the Retrieval Model.
        """

        def wrapper(self, *args, **kwargs):

            if not self.is_training:
                print("\033[33mModel is not suppose to do training.\033[0m")
                return

            function(self, *args, **kwargs)

        return wrapper

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

    @decorator
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

        # Load weights from checkpoint if exist.
        if self.hyperparameters["checkpoint_state"] != "no_checkpoint":
            checkpoint_status = self.checkpoint_manager.restore_or_initialize()
            if checkpoint_status != None:
                print("\033[33mModel restored from checkpoint.\033[0m")

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

                # Checkpoint the model.
                if self.hyperparameters["checkpoint_state"] == "epoch":
                    self.checkpoint_manager.save()

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

    @decorator
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

    def recommend(self, index):
        """
        Predict the model.
        """

        _, titles = self.index(np.array([index]))

        return titles

    # TODO: Try to decorate the index_from_dataset similar to the website.
    # TODO: If cant, try to implement with the exact way the website does it.
    def create_indexer(self):
        """
        Get the indexer.
        """
        self.index = tfrs.layers.factorized_top_k.BruteForce(self.query_tower)

        # Get the candidates.
        candidates = ds.get_candidates()

        self.index.index_from_dataset(tf.data.Dataset.zip(
            (candidates.batch(100), candidates.batch(100).map(self.candidate_tower))))

    def load(self, path=None):
        """
        Load the weights from saved model.
        """

        # If path is None, use the default path.
        if path is None:
            path = os.path.join(
                self.hyperparameters["save_dir"], self.hyperparameters["name"], self.id)

        # Load the model.
        self.index = tf.saved_model.load(path)

        # Log to user that the model was loaded using green color.
        print("\033[92mModel loaded from {}\033[0m".format(path))

    def save(self):
        """
        Save the model.
        """
        # Create the model directory.
        model_path = os.path.join(
            self.hyperparameters["save_dir"], self.hyperparameters["name"], self.id)

        # Create directory for the model.
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Get the indexer and save the indexer.
        # self.create_indexer()

        candidates = ds.get_candidates()

        index = tfrs.layers.factorized_top_k.BruteForce(self.query_tower)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            tf.data.Dataset.zip(
                (candidates.batch(100), candidates.batch(100).map(self.candidate_tower)))
        )

        # Save the indexer.
        tf.saved_model.save(index, model_path)

        # Log to user that the model was saved using green color.
        print("\033[92mModel saved to {}\033[0m".format(model_path))

    def add_checkpoint_manager(self):
        """
        Add the checkpoint manager.
        """
        # Set checkpoint.
        if self.hyperparameters["checkpoint_state"] != "no_checkpoint":
            self.checkpoint = tf.train.Checkpoint(
                model=self, optimizer=self.hyperparameters["optimizer"])
            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint, self.hyperparameters["checkpoint_dir"] + os.sep + self.hyperparameters["name"] + os.sep + self.id, max_to_keep=3)
