# This file will contains the configuration dictionary for the model.
from tensorflow import keras

# This is the configuration dictionary for the model.
config = {
    "name": "model",
    "input_shape": [28, 28, 1],
    "output_shape": 10,
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": keras.losses.CategoricalCrossentropy(),
    "metrics": [
        keras.metrics.CategoricalAccuracy()
    ],
    "batch_size": 32,
    "epochs": 10
}

model_definition = keras.Sequential([
    keras.layers.Flatten(input_shape=config["input_shape"]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(config["output_shape"], activation="softmax")
])