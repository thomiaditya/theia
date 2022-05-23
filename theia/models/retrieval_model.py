import os
import sys
import tensorflow as tf
import numpy as np
from ..config.recommender import retrieval_definition as rd, params, dataset as ds
from alive_progress import alive_bar


class RetrievalModel():
    def __init__(self):
        """
        Definition for the model definition in Retrieval Definition Class.
        """
        # Load the model definition from the config file.
        self.model = rd.RetrievalDefinition()

    def train(self):
        """
        Training the dataset from dataset file using alive progress bar.
        """
        # Prompt the user to enter the training data path with yellow color.
        pass

    def recommend(self, index):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
