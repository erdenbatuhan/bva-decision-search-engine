"""
 File:   word_embedding_featurizer.py
 Author: Batuhan Erden
"""

from src.featurization.tfidf_featurizer import TFIDFFeaturizer


class WordEmbeddingFeaturizer(TFIDFFeaturizer):

    def __init__(self, corpus, tokenization_segmenter):
        super().__init__(corpus, tokenization_segmenter)

    @staticmethod
    def add_additional_features(tfidf, spans):
        """
        Adds additional features to the feature vector (Overrides the inherited function)

        :param tfidf: Existing feature vector
        :param spans: Spans used to create the additional features
        :return: The new feature vector with the appended additional features
        """

        #         starts_normalized = np.array([s['start_normalized'] for s in spans])
        #         num_tokens = np.array([len(s['tokens_spacy']) for s in spans])
        #         np.concatenate((tfidf, np.expand_dims(starts_normalized, axis=1)), axis=1)

        return tfidf

