"""
 File:   tfidf_featurizer.py
 Author: Batuhan Erden
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.featurization.featurizer import Featurizer
from src.utils.logging_utils import log


class TfidfFeaturizer(Featurizer):

    MIN_TOKEN_FREQUENCY = 3
    NGRAM_RANGE = (1, 1)

    def __init__(self, corpus, tokenization_segmenter):
        super().__init__(corpus, tokenization_segmenter)
        self.tfidf_vectorizer = None

    def vectorize(self):
        """
        Vectorizes the training data
        """

        log("Initializing a TfidfVectorizer and fitting it on the training data..")

        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize,
                                                min_df=self.MIN_TOKEN_FREQUENCY,
                                                ngram_range=self.NGRAM_RANGE)

        train_span_texts = [span["txt"] for span in self.corpus.train_spans]
        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(train_span_texts)

        log("TfidfVectorizer successfully initialized!")

    def create_feature_vector(self, spans):
        """
        Creates the feature vector (Overrides the definition in the super class!)

        TfidfVectorizer features

        :param spans: Spans used to create the feature vector
        :return: The feature vector created
        """

        span_texts = [span["txt"] for span in spans]
        return self.tfidf_vectorizer.transform(span_texts).toarray()

    def __str__(self):
        # The feature names of the vectorizer
        return str(self.tfidf_vectorizer.get_feature_names())

