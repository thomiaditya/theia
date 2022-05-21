# Hyperparameters for the recommender system (Theia) would be defined here.

import tensorflow as tf

# Define the hyperparameters.
hyperparameters = {
    # Name of the recommender system. This will be the name of the project in wandb.
    'name': 'retrieval',
    # Unique id for the recommender system. This will be the name of the folder in wandb.
    'id': 'testing',
    'epochs': 3,  # Number of epochs to train the recommender system.
    'embedding_size': 32,  # Size of the embedding.

    'log_every': 1,  # Log the metrics every X batches.
    'train_batch_size': 8192,  # Batch size for training.
    'test_batch_size': 4096,  # Batch size for testing.

    # Optimizer for training.
    'optimizer': tf.keras.optimizers.Adagrad(learning_rate=0.1),
    'metrics_calculation': True,  # Calculate the metrics.

    # Wandb hyperparameters.
    'use_wandb': False,  # Use wandb.
}
