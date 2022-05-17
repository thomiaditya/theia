# This file will contains the configuration dictionary for the model.
from tensorflow import keras
import os

# This is the configuration dictionary for the model.
config = {
    "name": "mnist-handwriting",
    "input_shape": [28, 28, 1],
    "output_shape": 10,
    "use_wandb": True,
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": keras.losses.CategoricalCrossentropy(),
    "metrics": [
        keras.metrics.CategoricalAccuracy()
    ],
    "batch_size": 32,
    "epochs": 1,
    "callbacks": [],

    "checkpoint_state": "epoch", # "no_checkpoint" or "epoch"
    "checkpoint_dir": os.path.join(os.getcwd(), "history", "checkpoints"),
}

model_definition = keras.Sequential([
    keras.layers.Flatten(input_shape=config["input_shape"]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(config["output_shape"], activation="softmax")
])