"""
 File:   feature_generator.py
 Author: Batuhan Erden
"""

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from src.tokenizer import Tokenizer
from src.utils.logging_utils import log


class FeatureGenerator:

    MIN_TOKEN_FREQUENCY = 3
    NGRAM_RANGE = (1, 1)

    def __init__(self, corpus, tokenization_segmenter, embeddings_model):
        self.corpus = corpus
        self.tokenization_segmenter = tokenization_segmenter
        self.embeddings_model = embeddings_model

        # Add tokens to the spans for further featurization
        self.add_tokens_to_spans("train", self.corpus.train_spans)
        self.add_tokens_to_spans("val", self.corpus.val_spans)
        self.add_tokens_to_spans("test", self.corpus.test_spans)

        self.tfidf_vectorizer = None

    def add_tokens_to_spans(self, dataset_name, spans):
        """
        Adds tokens to spans for further use during featurization

        :param dataset_name: The name describing if it is the train, val or test data
        :param spans: Spans
        """

        log("Adding tokens to the %s spans.." % dataset_name)

        for span in tqdm(spans):
            span["tokens"] = self.tokenize(span["txt"])

        log("Tokens successfully added to the %s spans!" % dataset_name)

    def tokenize(self, plain_text):
        """
        Calls the tokenize function in Tokenizer and passes the specified segmenter

        :param plain_text: Plain text to be tokenized
        :return: Tokens generated using the specified segmenter and plain text
        """

        return Tokenizer.tokenize(self.tokenization_segmenter, plain_text)

    def vectorize(self):
        """
        Vectorizes the training data
        """

        log("Vectorizing the training data..")

        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize,
                                                min_df=self.MIN_TOKEN_FREQUENCY,
                                                ngram_range=self.NGRAM_RANGE)

        train_span_texts = [span["txt"] for span in self.corpus.train_spans]
        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(train_span_texts)

        log("The training data successfully vectorized!")

    def create_feature_expansions(self, spans):
        """
        Creates additional features to be used in expanding the feature vector

        Expansion 1: An average of the embedding vectors for the tokens in the sentence
                     (with the same dimension as the embedding model)
        Expansion 2: A single float variable with the [0-1] normalized position
                     of the sentence in the document
        Expansion 3: A single float variable representing the number of tokens in the sentence,
                     normalized by subtracting the mean and dividing by the standard deviation
                     across all sentence token counts

        :param spans: Spans used to create the additional features
        :return: The feature vector expansions
        """

        # Number of tokens across all sentences
        num_tokens = np.array([len(span["tokens"]) for span in spans])

        # Create the expansions
        return [
            # Expansion 1: An average of the embedding vectors
            np.array([
                np.mean([self.embeddings_model[token] for token in span["tokens"]], axis=0)
                for span in spans
            ]),
            # Expansion 2: Normalized positions
            np.expand_dims(np.array([span["start_normalized"] for span in spans]), axis=1),
            # Expansion 3: Number of tokens normalized
            np.expand_dims((num_tokens - np.mean(num_tokens)) / np.std(num_tokens), axis=1)
        ]

    def expand_feature_vector(self, dataset_name, tfidf, spans):
        """
        Expands the feature vector with additional features

        :param dataset_name: The name describing if it is the train, val or test data
        :param tfidf: Existing feature vector
        :param spans: Spans used to create the additional features
        :return: The new feature vector with the appended additional features
        """

        log("Expanding the feature vector in the %s dataset.." % dataset_name)

        shape_before_expansion = np.array(tfidf.shape)
        expansions = self.create_feature_expansions(spans)

        # Expand the feature vector
        for expansion in expansions:
            tfidf = np.concatenate((tfidf, expansion), axis=1)

        shape_after_expansion = np.array(tfidf.shape)
        num_new_features = sum([expansion.shape[1] for expansion in expansions])

        # Sanity-check the resulting shape
        assert (shape_before_expansion == shape_after_expansion - (0, num_new_features)).all(), \
               "Shapes do not match, expansion failed!"

        log("The feature vector successfully expanded in the %s dataset with %d additional features!" %
            (dataset_name, num_new_features))
        return tfidf

    def create_inputs_and_labels(self, feature_vector_expanded=False):
        """
        Creates X=inputs and y=labels for train, val and test sets

        :param feature_vector_expanded: Whether or not the feature vector gets expanded
                                        with additional features and embeddings (default: False)
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

        for dataset_name, spans in tqdm(spans_by_dataset.items()):
            # Transform
            tfidf = self.tfidf_vectorizer.transform([span["txt"] for span in spans]).toarray()

            # Expand feature vector
            if feature_vector_expanded:
                tfidf = self.expand_feature_vector(dataset_name, tfidf, spans)

            X[dataset_name] = tfidf
            y[dataset_name] = np.array([span["type"] for span in spans])

        log("The inputs and labels successfully created for train, val and test sets!")
        return X, y

    def __str__(self):
        # The feature names of the vectorizer
        return str(self.tfidf_vectorizer.get_feature_names())

