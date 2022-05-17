# This module contains the model class and its methods for training.
from audioop import add
import os
from turtle import title
import tensorflow as tf
import numpy as np
from theia.model import config
from alive_progress import alive_bar
from wandb import wandb
from datetime import datetime
from uuid import uuid4

class Model():
    def __init__(self):
        """
        Initialize the model class.
        """
        super(Model, self).__init__()

        self.config = config.config
        self.model = config.model_definition
        self.callbacks = tf.keras.callbacks.CallbackList(None, add_history=True)

    def train(self, train_dataset, val_dataset, use_wandb=False, log_wandb_on=100, log_on_epoch_end=True, checkpoint_on_epoch_end=False):
        """
        This method is called when the model is trained.
        """

        # Check if use wandb or not
        if use_wandb:
            wandb.init(project=self.config["name"])
        
        # Load all the callbacks.
        for callback in self.config["callbacks"]:
            self.callbacks.append(callback)

        # Check if checkpoint should be created.
        if checkpoint_on_epoch_end:
            self.checkpoint()
        
        # Append model to the callbacks list.
        self.callbacks.set_model(self.model)

        # Iterate over metrics to calculate the loss and accuracy.
        metrics_dict = {"loss": 0}
        metrics_string = ""
        for metric in self.config["metrics"]:
            metric.reset_states()
            metrics_string += "loss: {:.4f} {}: {:.4f} ".format(0, metric.name, metric.result())
            metrics_dict[metric.name] = metric.result()

        # Get the number of batches in the dataset.
        train_batches = len(train_dataset)
        val_batches = len(val_dataset)
        num_batches = train_batches + val_batches

        self.callbacks.on_train_begin(metrics_dict)

        # Print if the model is using wandb or not (print using yellow color).
        if use_wandb:
            print("\033[33mUsing Wandb, so the model will be logged to wandb and the model will be saved to wandb.\033[0m")
        else:
            print("\033[33mNot Using Wandb, so the model will not be logged to wandb and the model will not be saved to wandb.\033[0m")

        # Use the alive progress bar to show the progress of the training.
        with alive_bar(num_batches * self.config["epochs"], ctrl_c=False, manual=False, dual_line=True) as bar:

            # This is the training loop.
            for epoch in range(self.config["epochs"]):
                
                self.callbacks.on_epoch_begin(epoch, metrics_dict)

                # Setting the bar for each epoch.
                bar.title = "Epoch {}/{}".format(epoch + 1, self.config["epochs"])
                # counter = 1

                # Iterate over the training data.
                for batch, (images, labels) in enumerate(train_dataset):
                    
                    self.callbacks.on_batch_begin(batch, metrics_dict)
                    self.callbacks.on_train_batch_begin(batch, metrics_dict)

                    # Open Gradient Tape.
                    with tf.GradientTape() as tape:
                        # Compute the loss.
                        predictions = self.model(images, training=True)
                        loss = self.config["loss"](labels, predictions)

                    # Iterate over the metrics and update them.
                    for metric in self.config["metrics"]:
                        metric.update_state(labels, predictions)
                    
                    # Compute the gradients.
                    gradients = tape.gradient(loss, self.model.trainable_weights)
                    
                    # Apply the gradients.
                    self.config["optimizer"].apply_gradients(zip(gradients, self.model.trainable_weights))
                    
                    # Update the progress bar text.
                    bar.text = metrics_string

                    # Update metrics string with the new metrics.
                    if batch % 100 == 0:
                        metrics_string = "loss: {:.4f} ".format(loss.numpy())
                        metrics_dict["loss"] = loss.numpy()
                        for metric in self.config["metrics"]:
                            metrics_string += "{}: {:.4f} ".format(metric.name, metric.result())
                            metrics_dict[metric.name] = metric.result()

                    # Update the progress bar loading.
                    # bar(counter / num_batches)
                    # counter += 1
                    bar()

                    # If wandb is used, log the metrics.
                    if use_wandb and batch % log_wandb_on == 0 and not log_on_epoch_end:
                        wandb.log(metrics_dict)

                    self.callbacks.on_train_batch_end(batch, metrics_dict)
                    self.callbacks.on_batch_end(batch, metrics_dict)
                
                # Iterate over metrics to calculate the loss and accuracy.
                val_metrics_string = ""
                for metric in self.config["metrics"]:
                    metric.reset_states()
                    val_metrics_string += "val_loss: {:.4f} val_{}: {:.4f} ".format(0, metric.name, metric.result())
                
                # Iterate over the validation data.
                for batch, (images, labels) in enumerate(val_dataset):
                    
                    self.callbacks.on_batch_begin(batch, metrics_dict)
                    self.callbacks.on_test_batch_begin(batch, metrics_dict)

                    # Compute the loss.
                    predictions = self.model(images, training=False)
                    loss = self.config["loss"](labels, predictions)

                    # Calculate the metrics.
                    for metric in self.config["metrics"]:
                        metric.update_state(labels, predictions)

                    # Update the progress bar text.
                    bar.text = val_metrics_string

                    # Update the metrics string with the new metrics.
                    if batch % 100 == 0:
                        val_metrics_string  = "val_loss: {:.4f} ".format(loss.numpy())
                        metrics_dict["val_loss"] = loss.numpy()
                        for metric in self.config["metrics"]:
                            val_metrics_string += "val_{}: {:.4f} ".format(metric.name, metric.result())
                            metrics_dict["val_{}".format(metric.name)] = metric.result()

                    # Update the progress bar.
                    # bar(counter / num_batches)
                    # counter += 1
                    bar()

                    # If wandb is used, log the metrics.
                    if use_wandb and batch % log_wandb_on == 0 and not log_on_epoch_end:
                        wandb.log(metrics_dict)

                    self.callbacks.on_test_batch_end(batch, metrics_dict)
                    self.callbacks.on_batch_end(batch, metrics_dict)

                # Print the epoch and training metrics and validation metrics.
                if not use_wandb:
                    print("epoch {}: {} {}\n".format(epoch + 1, metrics_string, val_metrics_string))

                # If log_on_epoch_end is True, log the metrics.
                if use_wandb and log_on_epoch_end:
                    wandb.log(metrics_dict)

                self.callbacks.on_epoch_end(epoch, metrics_dict)
        
        self.callbacks.on_train_end(metrics_dict)

    def save(self, dir_path):
        """
        Save the model to the given path.
        """
        # Create id for the model combine with the current date and time.
        id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid4())[:8]

        # Create the model directory.
        model_path = os.path.join(dir_path, "saved_model", id)

        # Create directory for the model.
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save the model.
        self.model.save_weights(model_path + os.sep + "model.h5")

        # Log to user that the model was saved using yellow color.
        print("\033[33mModel saved to {}.\033[0m".format(model_path))

    def checkpoint(self):
        """
        Create a checkpoint callback to save the model every epoch.
        """
        # Get the path using id and the current date and time.
        id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid4())[:8]

        # Create the directory for the model.
        model_path = os.path.join(self.config["checkpoint_dir"], "checkpoints", id)

        # Create directory for the model.
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Create the checkpoint callback.
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, "{epoch:02d}-{val_loss:.2f}.hdf5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
            period=1
        )

        # Add the callback to the callbacks list.
        self.callbacks.append(checkpoint_callback)

        # Log to user that the checkpoint callback was added using yellow color.
        print("\033[33mCheckpoint callback added. The model will be saved every epoch in {}.\033[0m".format(self.config["checkpoint_dir"]))