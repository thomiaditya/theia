# This module contains the model class and its methods for training.
import os
import tensorflow as tf
import numpy as np
from ..config.classification import model_config, model_definition, load_datasets
from alive_progress import alive_bar
from wandb import wandb
from datetime import datetime
from uuid import uuid4


class ClassificationModel():
    def __init__(self):
        """
        Initialize the model class.
        """
        super().__init__()

        self.config = model_config
        self.model = model_definition
        self.callbacks = tf.keras.callbacks.CallbackList(
            None, add_history=True)

        # Set the model id.
        if self.config["id"] is None:
            self.id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + \
                "_" + str(uuid4())[:8]
        else:
            self.id = self.config["id"]

        # Set checkpoint.
        if self.config["checkpoint_state"] != "no_checkpoint":
            self.checkpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.config["optimizer"])
            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint, self.config["checkpoint_dir"] + os.sep + self.config["name"] + os.sep + self.id, max_to_keep=5)

        # Warning if model checkpoint is enabled and wandb is enabled, the model should be given an id, so that if it run the checkpoint will be resumed from the same run-id on wandb.
        if self.config["checkpoint_state"] != "no_checkpoint" and self.config["use_wandb"] and id is None:
            print("\033[93mWARNING: Checkpoint is enabled but no id was given. This will cause the model to be saved on wandb with the same run-id as the checkpoint.\033[0m")

        # Check if wandb is enabled and initialize the wandb object.
        self._check_wandb_and_initilize()

        # Prompt user that model was created using green color.
        print("\033[32mModel created: " + self.config["name"] + "\033[0m")

    def _check_wandb_and_initilize(self):
        """
        Check if wandb is enabled and initialize the wandb object.
        """

        # Check if wandb is enabled.
        if self.config["use_wandb"]:
            # Initialize the wandb object and set the project name. (Check if checkpoint_state is set to no_checkpoint and if not set the name into the id of the model.)
            if self.config["checkpoint_state"] != "no_checkpoint":
                self.wandb = wandb.init(
                    project=self.config["name"], id=self.id, resume="allow")

                print("\033[92mWandb initialized with project name: {} and id: {} USING checkpoint.\033[0m".format(
                    self.config["name"], self.id))
            else:
                self.wandb = wandb.init(project=self.config["name"])

                print("\033[92mWandb initialized with project name: {} and id: {} WITHOUT checkpoint.\033[0m".format(
                    self.config["name"], self.id))

    def wandb_log(self, logs, message=None):
        """
        Log the given logs to wandb.
        """
        # Log the given logs to wandb.
        if not self.config["use_wandb"]:
            return

        wandb.log(logs)
        print("\033[92mLogged to wandb.\033[0m")

    def _compile_metrics(self):
        """
        Compile the model with the given metrics.
        """
        # Compile the model.
        self.model.compile(
            optimizer=self.config["optimizer"],
            loss=self.config["loss"],
            metrics=self.config["metrics"]
        )

    def train(self, train_dataset=None, val_dataset=None):
        """
        This method is called when the model is trained.
        """

        # Load the train and validation datasets if they are not given.
        if train_dataset is None or val_dataset is None:
            train_dataset, val_dataset, info = load_datasets()

        # Check if use wandb or not
        use_wandb = self.config["use_wandb"]
        wandb = self.wandb if use_wandb else None

        # Load all the callbacks.
        for callback in self.config["callbacks"]:
            self.callbacks.append(callback)

        # Append model to the callbacks list.
        self.callbacks.set_model(self.model)

        # Restore the model from the checkpoint.
        if self.config["checkpoint_state"] != "no_checkpoint":
            checkpoint_status = self.checkpoint_manager.restore_or_initialize()
            if checkpoint_status != None:
                print("\033[33mModel restored from checkpoint.\033[0m")

        # This is the log for data collection.
        log_data = {
            "validation_data": val_dataset
        }

        # Add wandb to the log data.
        if use_wandb:
            log_data["wandb"] = wandb

        # Iterate over metrics to calculate the loss and accuracy.
        logs = {
            "loss": 0,
        }

        metrics_string = ""
        for metric in self.config["metrics"]:
            metric.reset_states()
            metrics_string += "loss: {:.4f} {}: {:.4f} ".format(
                0, metric.name, metric.result())
            logs[metric.name] = metric.result()

        # Get the number of batches in the dataset.
        train_batches = len(train_dataset)
        val_batches = len(val_dataset)
        num_batches = train_batches + val_batches

        self._compile_metrics()

        # Print if the model is using wandb or not (print using yellow color).
        if use_wandb:
            print(
                "\033[33mUsing Wandb, every logs will be redirected to wandb and will not be printed on the console.\033[0m")
        else:
            print(
                "\033[33mNot using Wandb, every logs will be printed to the console.\033[0m")

        # Update the log_data append with the logs and send it to ontrainbegin callback.
        log_data.update(logs)
        self.callbacks.on_train_begin(log_data)

        # Use the alive progress bar to show the progress of the training.
        with alive_bar(num_batches * self.config["epochs"], ctrl_c=False, manual=False, dual_line=True, spinner="pulse") as bar:

            # This is the training loop.
            for epoch in range(self.config["epochs"]):

                # Update the log_data append with the logs and send it to onepochbegin callback.
                log_data.update(logs)
                self.callbacks.on_epoch_begin(epoch + 1, log_data)

                # Setting the bar for each epoch.
                bar.title = "Epoch {}/{}".format(epoch + 1,
                                                 self.config["epochs"])

                # Iterate over the training data.
                for batch, (x, y) in enumerate(train_dataset):

                    self.callbacks.on_batch_begin(batch, logs)
                    self.callbacks.on_train_batch_begin(batch, logs)

                    loss, predictions = self._train_step(x, y)

                    # Update the progress bar text.
                    bar.text = metrics_string

                    # Update metrics string with the new metrics.
                    metrics_string = "loss: {:.4f} ".format(loss.numpy())
                    logs["loss"] = loss.numpy()
                    for metric in self.config["metrics"]:
                        metrics_string += "{}: {:.4f} ".format(
                            metric.name, metric.result())
                        logs[metric.name] = metric.result()

                    # Update the progress bar loading.
                    bar()

                    # If wandb is used, log the metrics.
                    # if use_wandb and batch % log_wandb_on == 0 and not log_on_epoch_end:
                    #     wandb.log(logs)

                    self.callbacks.on_train_batch_end(batch, logs)
                    self.callbacks.on_batch_end(batch, logs)

                # Iterate over metrics to calculate the loss and accuracy.
                val_metrics_string = ""
                for metric in self.config["metrics"]:
                    metric.reset_states()
                    val_metrics_string += "val_loss: {:.4f} val_{}: {:.4f} ".format(
                        0, metric.name, metric.result())

                # Iterate over the validation data.
                for batch, (x, y) in enumerate(val_dataset):

                    self.callbacks.on_batch_begin(batch, logs)
                    self.callbacks.on_test_batch_begin(batch, logs)

                    # Compute the loss and predictions.
                    loss, predictions = self._val_train_step(x, y)

                    # Update the progress bar text.
                    bar.text = val_metrics_string

                    # Update the metrics string with the new metrics.
                    val_metrics_string = "val_loss: {:.4f} ".format(
                        loss.numpy())
                    logs["val_loss"] = loss.numpy()
                    for metric in self.config["metrics"]:
                        val_metrics_string += "val_{}: {:.4f} ".format(
                            metric.name, metric.result())
                        logs["val_{}".format(metric.name)] = metric.result()

                    # Update the progress bar.
                    bar()

                    self.callbacks.on_test_batch_end(batch, logs)
                    self.callbacks.on_batch_end(batch, logs)

                # Print the epoch and training metrics and validation metrics.
                if not use_wandb:
                    print("epoch {}: {} | {}\n".format(
                        epoch + 1, metrics_string, val_metrics_string))

                # Save the model if the checkpoint_on_epoch_end is True.
                if self.config["checkpoint_state"] == "epoch":
                    self.checkpoint_manager.save()

                # If wandb is used, log the metrics.
                if use_wandb:
                    self.wandb.log(logs)

                log_data.update(logs)
                self.callbacks.on_epoch_end(epoch + 1, log_data)

        self.callbacks.on_train_end(log_data)

    def _train_step(self, x, y):
        """
        This method is used to train the model in a single step.
        """
        # Open Gradient Tape.
        with tf.GradientTape() as tape:
            # Compute the loss.
            predictions = self.model(x, training=True)
            loss = self.config["loss"](y, predictions)

        # Iterate over the metrics and update them.
        for metric in self.config["metrics"]:
            metric.update_state(y, predictions)

        # Compute the gradients.
        gradients = tape.gradient(loss, self.model.trainable_weights)

        # Apply the gradients.
        self.config["optimizer"].apply_gradients(
            zip(gradients, self.model.trainable_weights))

        return loss, predictions

    def _val_train_step(self, x, y):
        """
        This method is used to validate the model in a single step.
        """
        # Compute the loss.
        predictions = self.model(x, training=False)
        loss = self.config["loss"](y, predictions)

        # Calculate the metrics.
        for metric in self.config["metrics"]:
            metric.update_state(y, predictions)

        return loss, predictions

    def save(self):
        """
        Save the model to the given path.
        """

        # Create the model directory.
        model_path = os.path.join(
            self.config["save_dir"], self.config["name"], self.id)

        # Create directory for the model.
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save the model.
        self.model.save_weights(model_path + os.sep + "model.h5")

        # Log to user that the model was saved using green color.
        print("\033[92mModel saved to {}\033[0m".format(model_path))

    def _add_checkpoint_callback(self):
        """
        Create a checkpoint callback to save the model every epoch.
        """

        # Create the directory for the model.
        model_path = os.path.join(
            self.config["checkpoint_dir"], "checkpoints", self.id)

        # Create directory for the model.
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Create the checkpoint callback.
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                model_path, "{epoch:02d}-{val_loss:.2f}.hdf5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
            period=1
        )

        # Add the callback to the callbacks list.
        self.callbacks.append(checkpoint_callback)

        # Log to user that the checkpoint callback was added using yellow color.
        print("\033[33mCheckpoint callback added. The model will be saved every epoch in {}.\033[0m".format(
            self.config["checkpoint_dir"]))

    def predict(self, x, batch_size=None, verbose=False):
        """
        This method is used to predict the y for the given x.
        """
        # If batch_size is None, use the default batch size.
        if batch_size is None:
            batch_size = self.config["batch_size"]

        # Predict the y for the given x.
        predictions = self.model.predict(x)

        return predictions

    def load(self, path=None):
        """
        Load the model from the given path.
        """

        # If path is None, use the default path.
        if path is None:
            path = os.path.join(
                self.config["save_dir"], self.config["name"], self.id)

        # Load the model.
        self.model.load_weights(path + os.sep + "model.h5")

        # Log to user that the model was loaded using green color.
        print("\033[92mModel loaded from {}\033[0m".format(path))
