# This module contains the model class and its methods for training.
from audioop import add
import os
from turtle import title
import tensorflow as tf
import numpy as np
from model import config
from alive_progress import alive_bar
from wandb import wandb

class Model():
    def __init__(self):
        """
        Initialize the model class.
        """
        super(Model, self).__init__()

        self.config = config.config
        self.model = config.model_definition
        self.callbacks = tf.keras.callbacks.CallbackList(self.config["callbacks"], model=self.model, add_history=True)

    def train(self, train_dataset, val_dataset, use_wandb=False, log_wandb_on=100, log_on_epoch_end=True):
        """
        This method is called when the model is trained.
        """

        # Check if use wandb or not
        if use_wandb:
            wandb.init(project=self.config["name"])

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

        # Print if the model is using wandb or not.
        if use_wandb:
            print("Using wandb for logging, all logs will be saved to wandb/{}".format(self.config["name"]))
        else:
            print("Not using wandb, all logs will be printed to the console")

        # Use the alive progress bar to show the progress of the training.
        with alive_bar(num_batches * self.config["epochs"], ctrl_c=False, manual=False, enrich_print=False) as bar:

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

    def save(self, path):
        """
        Save the model to the given path.
        """
        self.model.save_weights(path)