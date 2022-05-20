# This file will contains the configuration dictionary for the model.
from tensorflow import keras
import os
from ...callbacks import WandbCallback

# This is the configuration dictionary for the model.
model_config = {
    # This is the name of the model. If use_wandb is True, this will be the name of the Project in wandb.
    "name": "mnist-handwriting",
    # This is the id of the model. If use_wandb is True, this will be the name of the Run in wandb.
    "id": "dense",
    "input_shape": [28, 28, 1],  # This is the shape of the input data.
    "output_shape": 10,  # Shape of the output data.
    "use_wandb": False,  # Set to True to use wandb
    # Optimizer to use.
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": keras.losses.CategoricalCrossentropy(),  # Loss function to use.
    "metrics": [  # Metrics to use. Use the keras.metrics.Metric class.
        keras.metrics.CategoricalAccuracy()
    ],
    "batch_size": 32,  # Batch size to use for training.
    "epochs": 2,  # Number of epochs to train for.
    "callbacks": [  # Callbacks to use. Use the keras.callbacks.Callback class.
        WandbCallback()
    ],
    # "no_checkpoint" or "epoch". If "epoch", the model will be saved after each epoch. If "no_checkpoint", the model will not be saved.
    "checkpoint_state": "epoch",
    # Directory to checkpoint the model to if checkpoint_state is "epoch".
    "checkpoint_dir": os.path.join(os.getcwd(), "history", "checkpoints"),
    # Directory to save the model.
    "save_dir": os.path.join(os.getcwd(), "history", "saved_models"),
}
