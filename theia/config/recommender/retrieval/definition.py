# Model definition ends here.

import tensorflow as tf
from .. import hyperparameters as hp
import tensorflow_recommenders as tfrs


def build_model(unique_queries, unique_candidates):
    """
    Build the model for the Retrieval Recommender System.
    """

    # Prompt the user that the model is building with yellow text.
    print("\033[93mBuilding the model...\033[0m")

    query_vocabs = tf.keras.layers.StringLookup(
        mask_token=None, name='user_id_lookup')
    query_vocabs.adapt(unique_queries, batch_size=10000)
    # Query tower is a model definition that will be used as a query for the retrieval model. For example, if we want to retrieve of the most related movie to the user, the user will be the query and the movie will be the item or we call the candidate.
    query_tower = tf.keras.Sequential([
        query_vocabs,
        tf.keras.layers.Embedding(
            input_dim=query_vocabs.vocabulary_size(),
            output_dim=hp.hyperparameters["embedding_size"],
            name='user_id_embedding'),
    ])

    candidate_vocabs = tf.keras.layers.StringLookup(
        mask_token=None, name='movie_title_lookup')
    candidate_vocabs.adapt(unique_candidates, batch_size=10000)
    # Item tower is a model definition that will be used as a item for the retrieval model. For example, if we want to retrieve of the most related movie to the user, the user will be the query and the movie will be the item or we call the candidate.
    candidate_tower = tf.keras.Sequential([
        candidate_vocabs,
        tf.keras.layers.Embedding(
            input_dim=candidate_vocabs.vocabulary_size(),
            output_dim=hp.hyperparameters["embedding_size"],
            name='movie_title_embedding'),
    ])

    # Prompt the user that the model is built with green text.
    print("\033[92mModel built.\033[0m")

    return query_tower, candidate_tower
