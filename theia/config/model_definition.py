from tensorflow import keras
import os
from .model_config import model_config as config

# Model definition goes here. Use the keras.layers.Layer class.
model_definition = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=config["input_shape"]),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(config["output_shape"], activation="softmax")
])