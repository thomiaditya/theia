# This file will contains the configuration dictionary for the model.
from tensorflow import keras

# This is the configuration dictionary for the model.
model_config = {
    "name": "model",
    "input_shape": [28, 28, 1],
    "output_shape": [10],
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": keras.losses.CategoryCrossentropy(),
    "metrics": [
        "accuracy"
    ],
    "batch_size": 32,
}