from random import sample
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])

timestamps = np.concatenate(
    list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)


def get_sample_input():
    """
    Get sample input.
    """

    sample_rating = next(iter(ratings.take(1)))
    sample_rating = {
        "movie_title": tf.reshape(sample_rating["movie_title"], (1,)),
        "user_id": tf.reshape(sample_rating["user_id"], (1,)),
        "timestamp": tf.reshape(sample_rating["timestamp"], (1,)),
    }

    return sample_rating


def get_train_data():
    """
    Get train data.
    """

    # Take the shuffled data and take the first 80,000.
    train = shuffled.take(80_000)

    return train


def get_eval_data():
    """
    Get test data.
    """

    # Get the last 20,000 ratings.
    test = shuffled.skip(80_000).take(20_000)

    return test


def get_candidate():
    """
    Get candidate.
    """
    return movies


def get_timestamps():
    """
    Get timestamps.
    """

    return timestamps, timestamp_buckets


def get_unique_query_candidate():
    """
    Get unique query and candidate.
    """

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    return unique_user_ids, unique_movie_titles
