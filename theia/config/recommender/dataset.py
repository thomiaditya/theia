# Dataset preprocessing will be on this file.

import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from .hyperparameters import hyperparameters as hp
from typing import Dict, Text, Tuple

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")

# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})

movies = movies.map(lambda x: x["movie_title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(
    100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000).shuffle(100_000).batch(hp["train_batch_size"])
test = shuffled.skip(80_000).take(20_000).batch(hp["test_batch_size"])


def get_candidates():
    """
    Get the candidates from the dataset. You will be implement to get the candidates from the dataset in this function.
    """

    return movies


def get_train_dataset() -> tf.data.Dataset:
    """
    Get the train dataset. You will be implement to get the train dataset in this function.
    """

    return train.cache()


def get_test_dataset() -> tf.data.Dataset:
    """
    Get the test dataset. You will be implement to get the test dataset in this function.
    """

    return test.cache()


def get_unique_vocabs():
    """
    Get the unique vocabs from the dataset. You will be implement to get the unique vocabs from the dataset in this function.
    """

    # movie_titles = movies.batch(1_000)
    # user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    # unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    # unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    # return unique_movie_titles, unique_user_ids

    return ratings.map(lambda x: x["user_id"]), movies


def compute_loss(model, features: Dict[Text, tf.Tensor], training) -> tf.Tensor:
    """
    Compute the loss for the Retrieval Recommender System. You will be implement to compute the loss in this function.
    """
    # Get the embedding of query and candidate.
    user_embeddings = model.query_tower(features["user_id"])
    movie_embeddings = model.candidate_tower(features["movie_title"])

    # Use model task to compute the loss.
    return model.hyperparameters["loss"](
        user_embeddings, movie_embeddings, compute_metrics=not training)
