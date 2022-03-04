"""
 File:   featurizer.py
 Author: Batuhan Erden
"""

import numpy as np
from abc import abstractmethod
from tqdm import tqdm

from src.tokenizer import Tokenizer
from src.utils.logging_utils import log


class Featurizer:

    def __init__(self, corpus, tokenization_segmenter):
        self.corpus = corpus
        self.tokenization_segmenter = tokenization_segmenter

    def tokenize(self, plain_text):
        """
        Calls the tokenize function in Tokenizer and passes the specified segmenter

        :param plain_text: Plain text to be tokenized
        :return: Tokens generated using the specified segmenter and plain text
        """

        return Tokenizer.tokenize(self.tokenization_segmenter, plain_text)

    def tokenize_spans(self, spans_by_dataset):
        """
        Tokenizes spans and store the tokens in each span for further use during featurization

        :param spans_by_dataset: Spans by each dataset (train, val and test)
        """

        for dataset_type, spans in spans_by_dataset.items():
            log("Adding tokens to the %s spans.." % dataset_type)

            for span in tqdm(spans):
                span["tokens"] = self.tokenize(span["txt"])

            log("Tokens successfully added to the %s spans!" % dataset_type)

    @abstractmethod
    def create_feature_vector(self, spans):
        raise Exception("This is an abstract method that needs to be overridden!")

    @staticmethod
    def create_feature_vector_expansions(spans):
        """
        Creates the additional features to be used in expanding the feature vector

        Expansion 1: A single float variable with the [0-1] normalized position
                     of the sentence in the document
        Expansion 2: A single float variable representing the number of tokens in the sentence,
                     normalized by subtracting the mean and dividing by the standard deviation
                     across all sentence token counts

        :param spans: Spans used to create the additional features
        :return: The feature vector expansions
        """

        # Number of tokens across all sentences
        num_tokens = np.array([len(span["tokens"]) for span in spans])

        # Create the expansions
        return [
            # Expansion 1: Normalized positions
            np.expand_dims(np.array([span["start_normalized"] for span in spans]), axis=1),
            # Expansion 2: Number of tokens normalized
            np.expand_dims((num_tokens - np.mean(num_tokens)) / np.std(num_tokens), axis=1)
        ]

    @staticmethod
    def expand_feature_vector(dataset_name, spans, feature_vector):
        """
        Expands the feature vector with additional features (@see create_feature_vector_expansions)

        :param dataset_name: The name describing if it is the train, val or test data
        :param spans: Spans used to create the additional features
        :param feature_vector: Existing feature vector
        :return: The new feature vector with the appended additional features
        """

        log("Expanding the feature vector (%s).." % dataset_name)

        shape_before_expansion = np.array(feature_vector.shape)
        expansions = Featurizer.create_feature_vector_expansions(spans)

        # Expand the feature vector
        for expansion in expansions:
            feature_vector = np.concatenate((feature_vector, expansion), axis=1)

        shape_after_expansion = np.array(feature_vector.shape)
        num_new_features = sum([expansion.shape[1] for expansion in expansions])

        # Sanity-check the resulting shape
        assert (shape_before_expansion == shape_after_expansion - (0, num_new_features)).all(), \
               "Shapes do not match, expansion failed!"

        log("The feature vector (%s) successfully expanded with %d additional features!" %
            (dataset_name, num_new_features))

        return feature_vector

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

        # Tokenize spans
        self.tokenize_spans(spans_by_dataset)

        for dataset_name, spans in tqdm(spans_by_dataset.items()):  # train, val and test
            # Create the feature vector
            feature_vector = self.create_feature_vector(spans)

            # Expand the feature vector
            feature_vector = self.expand_feature_vector(dataset_name, spans, feature_vector)

            # Create X=inputs and y=labels
            X[dataset_name] = feature_vector
            y[dataset_name] = np.array([span["type"] for span in spans])

        log("The inputs and labels successfully created for train, val and test sets!")
        return X, y

