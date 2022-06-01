# This is the hyperparameters of the model.

import os
from pickletools import optimize
import tensorflow as tf
import dotenv

# Load the environment variables to os.environ
dotenv.load_dotenv()

# Model settings
name = os.environ.get('MODEL_NAME', 'recommender')  # Name of the model
model_id = os.environ.get('MODEL_ID', '01234')  # ID of the model
checkpoint = True
# Directory to checkpoint the model to if checkpoint_state is "epoch".
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(
    os.getcwd(), ".history", "checkpoints"))
# Directory to save the model.
save_dir = os.environ.get("SAVE_DIR", os.path.join(
    os.getcwd(), ".history", "saved_models"))
use_wandb = os.environ.get("USE_WANDB", False)

# Model params
embedding_dimension = 32  # Spesify the embedding dimension of the models.
use_timestamps = True  # Whether to use timestamp.
layer_size = [64, 32]  # The layers of the model.

# Training params
compute_metrics_on_train = False  # If True, compute metrics on training set.
train_batch_size = 100  # Specify the batch size of the training.
eval_batch_size = 100  # Specify the batch size of the evaluation.
epochs = 1  # Specify the number of epochs.
# Specify the optimizer.
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
