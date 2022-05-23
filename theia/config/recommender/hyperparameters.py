# Hyperparameters for the recommender system (Theia) would be defined here.

import tensorflow as tf
import os

# Define the hyperparameters.
hyperparameters = {
    'mode': 'training',

    # Name of the recommender system. This will be the name of the project in wandb.
    'name': 'retrieval',
    # Unique id for the recommender system. This will be the name of the folder in wandb.
    'id': 'testing',
    'epochs': 3,  # Number of epochs to train the recommender system.
    'embedding_size': 64,  # Size of the embedding.

    'log_every': 1,  # Log the metrics every X batches.
    'train_batch_size': 4096,  # Batch size for training.
    'test_batch_size': 2000,  # Batch size for testing.

    # Optimizer for training.
    'optimizer': tf.keras.optimizers.Adagrad(learning_rate=0.5),
    'metrics_calculation': False,  # Calculate the metrics.

    "checkpoint_state": "no_checkpoint",
    # Directory to checkpoint the model to if checkpoint_state is "epoch".
    "checkpoint_dir": os.path.join(os.getcwd(), "history", "checkpoints"),
    # Directory to save the model.
    "save_dir": os.path.join(os.getcwd(), "history", "saved_models"),
    # Wandb hyperparameters.

    'use_wandb': False,  # Use wandb.
}
