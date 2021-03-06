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

    def tokenize_spans(self, dataset_name, spans):
        """
        Tokenizes spans and stores the tokens in each span for further use during featurization

        :param dataset_name: The name describing if it is the train, val or test data
        :param spans: Spans used to create the additional features
        """
        log("Adding tokens to the %s spans.." % dataset_name)

        for span in tqdm(spans):
            span["tokens"] = self.tokenize(span["txt"])

        log("Tokens successfully added to the %s spans!" % dataset_name)

    @abstractmethod
    def create_feature_vector(self, spans):
        raise Exception("This is an abstract method that needs to be overridden!")

    def create_feature_vector_expansions(self, spans):
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
        # Number of tokens across all sentences for given spans and train spans
        num_tokens = np.array([len(span["tokens"]) for span in spans])
        num_tokens_train = np.array([len(span["tokens"]) for span in self.corpus.train_spans]) \
            if self.corpus is not None else num_tokens

        # Create the expansions
        return [
            # Expansion 1: Normalized positions
            np.expand_dims(np.array([span["start_normalized"] for span in spans]), axis=1),
            # Expansion 2: Number of tokens normalized
            np.expand_dims((num_tokens - np.mean(num_tokens_train)) / np.std(num_tokens_train), axis=1)
        ]

    def expand_feature_vector(self, dataset_name, spans, feature_vector):
        """
        Expands the feature vector with additional features (@see create_feature_vector_expansions)

        :param dataset_name: The name describing if it is the train, val or test data
        :param spans: Spans used to create the additional features
        :param feature_vector: Existing feature vector
        :return: The new feature vector with the appended additional features
        """
        log("Expanding the feature vector (%s).." % dataset_name)

        shape_before_expansion = np.array(feature_vector.shape)
        expansions = self.create_feature_vector_expansions(spans)

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

    def create_inputs_and_labels_for_spans(self, dataset_name, spans, tokenize=False):
        """
        Creates X=inputs and y=labels for given spans

        :param dataset_name: The name describing if it is the train, val or test data
        :param spans: Spans used to create the additional features
        :param tokenize: When set to True, the spans will be tokenized - used during testing,
                         see "analyze.py" (default: False)
        :return: X and y for given spans
        """
        # Tokenize spans (Only used during testing, see "analyze.py")
        if tokenize:
            self.tokenize_spans(dataset_name, spans)

        # Create the feature vector
        feature_vector = self.create_feature_vector(spans)

        # Expand the feature vector
        feature_vector = self.expand_feature_vector(dataset_name, spans, feature_vector)

        # Create X=inputs and y=labels
        return feature_vector, np.array([span["type"] for span in spans])

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

        for dataset_name, spans in spans_by_dataset.items():  # train, val and test
            # Tokenize spans
            self.tokenize_spans(dataset_name, spans)

            # Create X and y
            X[dataset_name], y[dataset_name] = self.create_inputs_and_labels_for_spans(dataset_name, spans)

        log("The inputs and labels successfully created for train, val and test sets!")
        return X, y

    def analyze_shapes(self, X, y):
        """
        Analyzes the shapes of the inputs and labels

        :param X: Inputs
        :param y: Labels
        :return Shape analysis output
        """
        shape_analysis_output_lines = [f"{type(self).__name__} Shapes"]

        for dataset_type in X:
            shape_analysis_output_lines.append(
                f"- The shapes of X_{dataset_type} and y_{dataset_type} are " +
                f"{X[dataset_type].shape} and {y[dataset_type].shape} respectively."
            )

        return "\n".join(shape_analysis_output_lines)
