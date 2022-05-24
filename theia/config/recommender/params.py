# This is the hyperparameters of the model.

from pickletools import optimize
import tensorflow as tf


embedding_dimension = 32  # Spesify the embedding dimension of the models.
train_batch_size = 100  # Specify the batch size of the training.
epochs = 1  # Specify the number of epochs.
# Specify the optimizer.
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
