# This is the hyperparameters of the model.

from pickletools import optimize
import tensorflow as tf

# Model params
embedding_dimension = 32  # Spesify the embedding dimension of the models.

# Training params
compute_metrics_on_train = False  # If True, compute metrics on training set.
train_batch_size = 100  # Specify the batch size of the training.
eval_batch_size = 100  # Specify the batch size of the evaluation.
epochs = 1  # Specify the number of epochs.
# Specify the optimizer.
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
