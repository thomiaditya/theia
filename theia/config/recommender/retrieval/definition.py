# Model definition ends here.

import tensorflow as tf
from .. import hyperparameters as hp
import tensorflow_recommenders as tfrs


def build_model(unique_queries, unique_candidates):
    # Query tower is a model definition that will be used as a query for the retrieval model. For example, if we want to retrieve of the most related movie to the user, the user will be the query and the movie will be the item or we call the candidate.
    query_tower = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_queries, mask_token=None, name='user_id_lookup'),
        tf.keras.layers.Embedding(
            input_dim=len(unique_queries),
            output_dim=hp.hyperparameters["embedding_size"],
            name='user_id_embedding'),
    ])

    # Item tower is a model definition that will be used as a item for the retrieval model. For example, if we want to retrieve of the most related movie to the user, the user will be the query and the movie will be the item or we call the candidate.
    candidate_tower = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_candidates, mask_token=None, name='movie_title_lookup'),
        tf.keras.layers.Embedding(
            input_dim=len(unique_candidates),
            output_dim=hp.hyperparameters["embedding_size"],
            name='movie_title_embedding'),
    ])

    return query_tower, candidate_tower
