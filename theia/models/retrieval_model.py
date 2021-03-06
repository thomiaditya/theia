import os
import sys
from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from ..config.recommender import params, gcs_utils as up
from ..utils import Logger
from alive_progress import alive_bar
import wandb
import dotenv

dotenv.load_dotenv()

class RetrievalModel():
    def __init__(self, epochs=None):
        """
        Definition for the model definition in Retrieval Definition Class.
        """
        from ..config.recommender import dataset as ds, retrieval_definition as rd

        self.ds = ds

        # Load the model definition from the config file.
        self.model = rd.RetrievalDefinition()
        self.logger = Logger()

        if epochs is not None:
            params.epochs = epochs

        # Prompt the user that the model is created.
        self.logger.write(
            "Model {} is created successfully".format(self), level="INFO")

    def _check_and_initialize_wandb(self):
        """
        Check and initialize the wandb.
        """
        # Check if the model use wandb.
        if params.use_wandb:
            # Initialize the wandb.
            _wandb = wandb.init(project=params.name,
                                id=params.model_id, resume="allow", config={
                                    "epochs": params.epochs,
                                    "train_batch_size": params.train_batch_size,
                                    "eval_batch_size": params.eval_batch_size,
                                })

            self.logger.write(
                "Wandb is initialized.", level="WARNING")

            return _wandb

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
        # Check and initialize the wandb.
        self.wandb = self._check_and_initialize_wandb()

        # Prompt the user that the training is starting.
        self.logger.write(
            "Training {} is starting...".format(self.model), level="WARNING")

        # Get the train data.
        train_data = self.ds.get_train_data()
        train_data = train_data.batch(params.train_batch_size)

        # Get the validation data.
        val_data = self.ds.get_eval_data()
        val_data = val_data.batch(params.eval_batch_size)

        # Get the number of batches.
        num_batches = len(list(train_data)) + len(list(val_data))

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

                # Compute the metrics on the validation data.
                for batch, features in enumerate(val_data):
                    val_metrics, val_logs = self.eval_step(features, True)

                    # Metrics to string.
                    metrics_string = self.metrics_to_string(val_metrics)

                    # Update the alive progress bar text.
                    bar.text = metrics_string

                    # Update the progress bar.
                    bar()

                # Log to wandb.
                if params.use_wandb:
                    wandb.log({**metrics, **val_logs})
                else:
                    # Print the result metrics and epoch.
                    print("epoch {}: {}".format(
                        epoch + 1, self.metrics_to_string({**metrics})))

                # Save the checkpoint.
                if params.checkpoint:
                    self.checkpoint_manager.save()
        
        # Upload the model to GCS.
        print("Uploading the model to GCS...")
        up.upload_fileobj(
            project=os.environ.get("GOOGLE_PROJECT_ID", "zeta-resource-351216"),
            bucket=os.environ.get("GOOGLE_BUCKET_NAME", "theia-recommender"),
            directory_to_upload=os.path.expanduser("~/.history"),
            destination_path=".history",
            service_account_credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None),
        )

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
        metrics = {"loss": loss, "total_loss": total_loss}

        # if params.compute_metrics_on_train:
        #     metrics["factorized_top_k"] = np.array(
        #         [metric.result() for metric in self.model.metrics])

        return metrics

    def eval_step(self, features, training=False):
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
        metrics = {"val_loss": loss, "val_total_loss": total_loss}

        factorize_top_k = {metric.name: metric.result()
                           for metric in self.model.metrics}
        logs = {**metrics, **factorize_top_k}

        if not training or (training and params.compute_metrics_on_train):
            metrics["val_factorized_top_k"] = np.array(
                [metric.result() for metric in self.model.metrics])

        return metrics, logs

    def evaluate(self, eval_data):
        """
        Evaluation the dataset from dataset file.
        """
        # Prompt the user that the evaluation is starting.
        self.logger.write(
            "Evaluation {} is starting...".format(self.model), level="WARNING")

        # Get the eval data.
        eval_data = self.ds.get_eval_data()
        eval_data = eval_data.batch(params.eval_batch_size)

        num_batches = len(eval_data)

        with alive_bar(num_batches, ctrl_c=False, manual=False, dual_line=True, spinner="pulse", enrich_print=False) as bar:

            # Set the title to evaluation.
            bar.title = "Evaluation"

            # Iterate over the batches.
            for batch, features in enumerate(eval_data):

                # Compute the loss.
                metrics, logs = self.eval_step(features)

                # Metrics to string.
                metrics_string = self.metrics_to_string(metrics)

                # Update the alive progress bar text.
                bar.text = metrics_string

                # Update the progress bar.
                bar()

                # Log to wandb.
                if params.use_wandb:
                    if batch % (num_batches // 100) == 0:
                        wandb.log(logs)

            # Print the result metrics.
            print("eval: {}".format(metrics_string))

    def create_indexer(self):
        """
        Create the indexer.
        """
        self.indexer = tfrs.layers.factorized_top_k.BruteForce(
            self.model.query_model)

        candidates = self.ds.get_candidate()

        # Recommends candidate of all the candidates.
        self.indexer.index_from_dataset(tf.data.Dataset.zip(
            (candidates.batch(100), candidates.batch(100).map(self.model.candidate_model))))

        # Call the indexer to build the index.
        self.indexer(self.ds.get_sample_input())

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
        _, candidates = self.indexer({
            "user_id": tf.constant([index]),
        })

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
        """
        Recommend the candidates.
        """
        # Load the model.
        # Check if path is None.
        if path is None:
            path = os.path.join(
                params.save_dir, params.name, params.model_id)

        # Load the indexer.
        indexer = tf.saved_model.load(path)

        # Recommend the candidates.
        _, candidates = indexer({
            "user_id": tf.constant([index]),
        })

        # Return the candidates.
        return candidates
