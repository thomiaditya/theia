# This file will contains the configuration dictionary for the model.
from tensorflow import keras
import os

# This is the configuration dictionary for the model.
config = {
    "name": "mnist-handwriting",
    "id": "convolutional-8757a65f1f124bd794730308d94f3d48",
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
    "save_dir": os.path.join(os.getcwd(), "history", "saved_models"),
}

model_definition = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=config["input_shape"]),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(config["output_shape"], activation="softmax")
])