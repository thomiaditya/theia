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
})
movies = movies.map(lambda x: x["movie_title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)


def get_train_data():
    """
    Get train data.
    """

    # Take the shuffled data and take the first 80,000.
    train = shuffled.take(80_000)

    return train


def get_test_data():
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


def get_unique_query_candidate():
    """
    Get unique query and candidate.
    """

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    return unique_user_ids, unique_movie_titles
