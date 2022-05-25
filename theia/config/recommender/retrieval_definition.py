# This is the model definition of retrieval model.

from typing import List, Dict, Tuple, Text
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from . import dataset as ds, params

# Get unique query and candidate and timestamp.
unique_user_ids, unique_movie_titles = ds.get_unique_query_candidate()
timestamps, timestamps_buckets = ds.get_timestamps()

# Get candidate from dataset.
candidate = ds.get_candidate()

embedding_dimension = params.embedding_dimension


class UserModel(tf.keras.Model):
    def __init__(self, use_timestamps=False):
        super(UserModel, self).__init__()
        self.use_timestamps = use_timestamps
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            # We add an additional embedding to account for unknown tokens.
            tf.keras.layers.Embedding(
                len(unique_user_ids) + 1, embedding_dimension)
        ])

        if self.use_timestamps:
            self.timestamp_model = tf.keras.Sequential([
                tf.keras.layers.Discretization(timestamps_buckets.tolist()),
                tf.keras.layers.Embedding(
                    len(timestamps_buckets) + 1, embedding_dimension)
            ])

            self.normalized_timestamp = tf.keras.layers.Normalization(
                axis=None
            )

            self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        if not self.use_timestamps:
            return self.user_model(inputs["user_id"])

        return tf.concat([
            self.user_model(inputs["user_id"]),
            self.timestamp_model(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), [-1, 1])
        ], axis=1)


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(
                layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class MovieModel(tf.keras.Model):
    def __init__(self):
        super(MovieModel, self).__init__()

        max_token = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(
                len(unique_movie_titles) + 1, embedding_dimension)
        ])

        self.title_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_token
        )

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(
                max_token, embedding_dimension, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])

        self.title_vectorizer.adapt(candidate)

    def call(self, inputs):
        return tf.concat([
            self.title_embedding(inputs),
            self.title_text_embedding(inputs)
        ], axis=1)


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = MovieModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(
                layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class RetrievalDefinition(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.query_model: tf.keras.Model = QueryModel([embedding_dimension])
        self.candidate_model: tf.keras.Model = CandidateModel(
            [embedding_dimension])
        self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate.batch(128).map(self.candidate_model)
            )
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"]
        })

        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.candidate_model(
            features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=not training)
