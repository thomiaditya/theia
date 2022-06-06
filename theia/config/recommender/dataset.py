from random import sample
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from . import gcs_utils as dl
import dotenv

# Load the environment variables to os.environ
dotenv.load_dotenv()

credential = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)

ratings = pd.read_csv(dl.get_byte_fileobj('zeta-resource-351216', 'theia-recommender', 'data/ratings.csv', credential))
therapists = pd.read_csv(dl.get_byte_fileobj('zeta-resource-351216', 'theia-recommender', 'data/therapists.csv', credential))

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
        "user_id": tf.reshape(sample_rating["user_id"], (1,)),
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
