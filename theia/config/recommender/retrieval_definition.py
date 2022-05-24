# This is the model definition of retrieval model.

from typing import List, Dict, Tuple, Text
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from . import dataset as ds, params

# Get unique query and candidate.
unique_user_ids, unique_movie_titles = ds.get_unique_query_candidate()

# Get candidate from dataset.
candidate = ds.get_candidate()

embedding_dimension = params.embedding_dimension

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
    tf.keras.layers.Embedding(
        len(unique_movie_titles) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates=candidate.batch(128).map(movie_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)


class RetrievalDefinition(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])

        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=not training)
