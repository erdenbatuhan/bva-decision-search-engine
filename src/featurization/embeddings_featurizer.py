"""
File:   embeddings_featurizer.py
Author: Batuhan Erden
"""

import numpy as np

from src.featurization.featurizer import Featurizer


class EmbeddingsFeaturizer(Featurizer):

    def __init__(self, corpus, tokenization_segmenter, embeddings_model):
        super().__init__(corpus, tokenization_segmenter)
        self.embeddings_model = embeddings_model

    def create_feature_vector(self, spans):
        """
        Creates the feature vector (Overrides the definition in the super class!)

        Feature vector gets created using the average of the embedding vectors for the tokens in the sentence
        (with the same dimension as the embedding model)

        :param spans: Spans used to create the feature vector
        :return: The feature vector created
        """
        return np.array([
            np.mean([self.embeddings_model[token] for token in span["tokens"]], axis=0)
            for span in spans
        ])
