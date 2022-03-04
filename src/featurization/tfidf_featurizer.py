"""
 File:   tfidf_featurizer.py
 Author: Batuhan Erden
"""

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from src.tokenizer import Tokenizer
from src.utils.logging_utils import log


class TFIDFFeaturizer:

    MIN_TOKEN_FREQUENCY = 3
    NGRAM_RANGE = (1, 1)

    def __init__(self, corpus, tokenization_segmenter):
        self.corpus = corpus
        self.tokenization_segmenter = tokenization_segmenter

        self.tfidf_vectorizer = None

    def tokenize(self, plain_text):
        """
        Calls the tokenize function in Tokenizer and passes the specified segmenter

        :param plain_text: Plain text to be tokenized
        :return: Tokens generated using the specified segmenter and plain text
        """

        return Tokenizer.tokenize(self.tokenization_segmenter, plain_text)

    def fit(self):
        """
        Vectorizes the training data
        """

        log("Vectorizing the training data using TfidfVectorizer..")

        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize,
                                                min_df=self.MIN_TOKEN_FREQUENCY,
                                                ngram_range=self.NGRAM_RANGE)

        train_span_texts = [span['txt'] for span in self.corpus.train_spans]
        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(train_span_texts)

        log("The training data vectorized successfully using TfidfVectorizer!")

    def get_vectorizer_feature_names(self):
        """
        Returns the feature names of the vectorizer

        :return: The feature names of the vectorizer
        """

        return self.tfidf_vectorizer.get_feature_names()

    @staticmethod
    def add_additional_features(tfidf, spans):
        """
        Adds additional features to the feature vector

        :param tfidf: Existing feature vector
        :param spans: Spans used to create the additional features
        :return: The new feature vector with the appended additional features
        """

        return tfidf

    def create_inputs_and_labels(self):
        """
        Creates X=inputs and y=labels for train, val and test sets

        :return: X and y for train, val and test sets
        """

        log("Creating the inputs and labels for train, val and test sets..")

        spans_by_dataset = {
            "train": self.corpus.train_spans,
            "val": self.corpus.val_spans,
            "test": self.corpus.test_spans
        }

        X = {"train": None, "val": None, "test": None}
        y = {"train": None, "val": None, "test": None}

        for dataset_key, spans in tqdm(spans_by_dataset.items()):
            # Transform
            tfidf = self.tfidf_vectorizer.transform([span['txt'] for span in spans]).toarray()

            # Add additional features
            tfidf = self.add_additional_features(tfidf, spans)

            X[dataset_key] = tfidf
            y[dataset_key] = np.array([span['type'] for span in spans])

        log("The inputs and labels successfully created for train, val and test sets!")
        return X, y

