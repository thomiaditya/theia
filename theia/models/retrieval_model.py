import os
import sys
from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs
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
            "Model {} is created successfully".format(self), level="INFO")

    def _checkpoint_decorator(func):
        """
        Checkpoint decorator for the model.
        """

        def wrapper(self, *args, **kwargs):
            if params.checkpoint:
                checkpoint = tf.train.Checkpoint(
                    model=self.model, optimizer=params.optimizer)

                # Create checkpoint manager.
                self.checkpoint_manager = tf.train.CheckpointManager(
                    checkpoint, directory=os.path.join(
                        params.checkpoint_dir, params.name, params.model_id),
                    max_to_keep=3)

                # Intialize the checkpoint or restore it.
                status = self.checkpoint_manager.restore_or_initialize()
                if status:
                    self.logger.write(
                        "Checkpoint {} is restored successfully".format(
                            checkpoint),
                        level="INFO")

            func(self, *args, **kwargs)

        return wrapper

    @_checkpoint_decorator
    def train(self):
        """
        Training the dataset from dataset file using alive progress bar.
        """
        # Prompt the user that the training is starting.
        self.logger.write(
            "Training {} is starting...".format(self.model), level="WARNING")

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

                # Save the checkpoint.
                if params.checkpoint:
                    self.checkpoint_manager.save()

        # Evaluate the model.
        self.evaluate()

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
            loss = self.model.compute_loss(
                features, training=not params.compute_metrics_on_train)

            # Handle regularization losses as well.
            regularization_loss = sum(self.model.losses)

            total_loss = loss + regularization_loss

        # Update the weights.
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        params.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        # Return the metrics.
        metrics = {}

        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        if params.compute_metrics_on_train:
            metrics["factorized_top_k"] = np.array(
                [metric.result() for metric in self.model.metrics])

        return metrics

    def eval_step(self, features):
        """
        Evaluation step for the model.
        """
        # Loss computation.
        loss = self.model.compute_loss(
            features, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.model.losses)

        total_loss = loss + regularization_loss

        # Return the metrics.
        metrics = {}

        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        metrics["factorized_top_k"] = np.array(
            [metric.result() for metric in self.model.metrics])

        return metrics

    def evaluate(self):
        """
        Evaluation the dataset from dataset file.
        """
        # Prompt the user that the evaluation is starting.
        self.logger.write(
            "Evaluation {} is starting...".format(self.model), level="WARNING")

        # Get the eval data.
        eval_data = ds.get_eval_data()
        eval_data = eval_data.batch(params.eval_batch_size)

        with alive_bar(len(eval_data), ctrl_c=False, manual=False, dual_line=True, spinner="pulse", enrich_print=False) as bar:

            # Set the title to evaluation.
            bar.title = "Evaluation"

            # Iterate over the batches.
            for batch, features in enumerate(eval_data):

                # Compute the loss.
                metrics = self.eval_step(features)

                # Metrics to string.
                metrics_string = self.metrics_to_string(metrics)

                # Update the alive progress bar text.
                bar.text = metrics_string

                # Update the progress bar.
                bar()

            # Print the result metrics.
            print("eval: {}".format(metrics_string))

    def create_indexer(self):
        """
        Create the indexer.
        """
        self.indexer = tfrs.layers.factorized_top_k.BruteForce(
            self.model.query_model)

        candidates = ds.get_candidate()

        # Recommends candidate of all the candidates.
        self.indexer.index_from_dataset(tf.data.Dataset.zip(
            (candidates.batch(100), candidates.batch(100).map(self.model.candidate_model))))

    def _indexer_decorator(func):
        """
        Decorator for the indexer.
        """

        def wrapper(self, index, load_model_dir=None, *args, **kwargs):
            # Create brute force layer.
            if load_model_dir == None:
                # Create the indexer.
                if not hasattr(self, "indexer"):
                    self.create_indexer()

                # Call the function.
                return func(self, index, *args, **kwargs)

            if load_model_dir == "last_saved":
                # get last saved model dir.
                self.indexer = self.load()

                # Call the function.
                return func(self, index, *args, **kwargs)

        return wrapper

    @_indexer_decorator
    def recommend(self, index):
        """
        Recommend the candidates.
        """
        # Recommend the candidates.
        _, candidates = self.indexer(tf.constant([index]))

        # Return the candidates.
        return candidates

    def save(self, path=None):
        """
        Save the model.
        """
        # Check the indexer.
        if not hasattr(self, "indexer"):
            self.create_indexer()

        # If path is None, use the default path.
        if path is None:
            path = os.path.join(
                params.save_dir, params.name, params.model_id)

        # Save the indexer.
        tf.saved_model.save(self.indexer, path)

        # Log the save.
        self.logger.write("Model saved to {}".format(path), level="INFO")

    def load(self, path=None):
        """
        Load the model.
        """
        # Check if path is None.
        if path is None:
            path = os.path.join(
                params.save_dir, params.name, params.model_id)

        # Load the indexer.
        indexer = tf.saved_model.load(path)

        # Log the load.
        self.logger.write("Model loaded from {}".format(path), level="INFO")

        return indexer

    @staticmethod
    def static_recommend(index, path=None):
        pass
