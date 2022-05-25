# This is the hyperparameters of the model.

import os
from pickletools import optimize
import tensorflow as tf

# Model settings
name = "retrieval_model"  # Name of the model
model_id = "546313-test"  # Model id
checkpoint = True
# Directory to checkpoint the model to if checkpoint_state is "epoch".
checkpoint_dir = os.path.join(os.getcwd(), "history", "checkpoints")
# Directory to save the model.
save_dir = os.path.join(os.getcwd(), "history", "saved_models")
use_wandb = False  # Whether to use wandb for logging.

# Model params
embedding_dimension = 32  # Spesify the embedding dimension of the models.
use_timestamps = True  # Whether to use timestamp.

# Training params
compute_metrics_on_train = False  # If True, compute metrics on training set.
train_batch_size = 100  # Specify the batch size of the training.
eval_batch_size = 100  # Specify the batch size of the evaluation.
epochs = 1  # Specify the number of epochs.
# Specify the optimizer.
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
