from random import sample
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import urllib.request
import os
import pandas as pd


def get_file(filename, url):
    """
    Get file.
    """

    directory = os.path.join(os.path.expanduser("~"), ".data")
    filepath = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)

    return filepath


THERAPISTS_DATA_URL = "https://res.cloudinary.com/dhuvbrmgg/raw/upload/v1653642312/Data/therapists.csv"
RATINGS_DATA_URL = "https://res.cloudinary.com/dhuvbrmgg/raw/upload/v1653642316/Data/ratings.csv"

ratings_filepath = get_file("ratings.csv", RATINGS_DATA_URL)
therapists_filepath = get_file(
    "therapists.csv", THERAPISTS_DATA_URL)

ratings = pd.read_csv(ratings_filepath)
therapists = pd.read_csv(therapists_filepath)

ratings = tf.data.Dataset.from_tensor_slices(dict(ratings))
therapists = tf.data.Dataset.from_tensor_slices(dict(therapists))

ratings = ratings.map(lambda x: {
    "user_id": tf.strings.as_string(x["id_user"]),
    "therapist_id": tf.strings.as_string(x["id_therapist"]),
})

therapists = therapists.map(lambda x: tf.strings.as_string(x["id"]))

tf.random.set_seed(42)
shuffled = ratings.shuffle(500, seed=42, reshuffle_each_iteration=False)


def get_sample_input():
    """
    Get sample input.
    """

    sample_rating = next(iter(ratings.take(1)))
    sample_rating = {
        "therapist_id": tf.reshape(sample_rating["therapist_id"], (1,)),
        "user_id": tf.reshape(sample_rating["user_id"], (1,)),
    }

    return sample_rating


def get_train_data():
    """
    Get train data.
    """

    # Take the shuffled data and take the first 80,000.
    train = shuffled.take(80000)

    return train


def get_eval_data():
    """
    Get test data.
    """

    # Get the last 20,000 ratings.
    test = shuffled.skip(80000).take(20000)

    return test


def get_candidate():
    """
    Get candidate.
    """
    return therapists


def get_timestamps():
    """
    Get timestamps.
    """

    return None, None


def get_unique_query_candidate():
    """
    Get unique query and candidate.
    """

    therapist_id = therapists.batch(1000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_therapist_id = np.unique(np.concatenate(list(therapist_id)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    return unique_user_ids, unique_therapist_id


# print(get_sample_input())
# print(get_candidate())
