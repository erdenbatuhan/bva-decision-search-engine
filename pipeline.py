"""
 File:   pipeline.py
 Author: Batuhan Erden
"""

import sys

from src.corpus import Corpus

from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter

from src.unlabeled_tokenizer import UnlabeledTokenizer


def initialize_corpus(annotations_filepath, unlabeled_data_dir):
    """
    Step 1: Corpus Initialization & Dataset Splitting

    :param annotations_filepath: Path to the file containing the annotations
    :param unlabeled_data_dir: Directory containing the unlabeled data
    :return Corpus initialized
    """

    # Initialize the Corpus
    corpus = Corpus(annotations_filepath, unlabeled_data_dir)
    print(corpus)

    # Return the Corpus
    return corpus


def initialize_segmenters(corpus, debug=False):
    """
    Step 2: Sentence Segmentation

    :param corpus: Corpus
    :param debug: Whether or not an error analysis is performed for the segmentations (default: False)
    :return: segmenters
    """

    # Initialize the segmenters
    segmenters = {
        # Step 2.1: Standard segmentation analysis
        "NaiveSpacySegmenter": SpacySegmenter(corpus=corpus),
        # Step 2.2: Improved segmentation analysis
        "ImprovedSpacySegmenter": SpacySegmenter(corpus=corpus, improved=True),
        # Step 2.3: Luima (A law-specific sentence segmenter)
        "LuimaLawSegmenter": LuimaLawSegmenter(corpus=corpus)
    }

    # Apply each segmentation and analyze the errors
    if debug:
        for segmenter in segmenters.values():
            segmenter.apply_segmentation(annotated=True, debug=True)

    return segmenters


def preprocess_data(segmenters, generate_new=False):
    """
    Step 3: Preprocessing (Tokenization)

    :param segmenters: Segmenters initialized
    :param generate_new: When set to True, sentences and tokens are generated from scratch. Otherwise, they are loaded.
    """

    # Initialize the unlabeled tokenizer
    unlabeled_tokenizer = UnlabeledTokenizer(segmenters)

    # Generate or load sentences and tokens
    if generate_new:
        return unlabeled_tokenizer.generate()  # Takes about 4 to 6 hours..
    else:
        return unlabeled_tokenizer.load()


def train_word_embeddings():
    """
    Step 4: Developing Word Embeddings
    """

    pass


def train_classifiers():
    """
    Step 5: Training Classifiers
    """

    pass


def analyze_errors():
    """
    Step 6: Error Analysis
    """

    pass


def train(annotations_filepath, unlabeled_data_dir):
    """
    Trains the best model possible for BVA decisions

    :param annotations_filepath: Path to the file containing the annotations
    :param unlabeled_data_dir: Directory containing the unlabeled data
    """

    corpus = initialize_corpus(annotations_filepath, unlabeled_data_dir)    # Step 1: Dataset Splitting
    segmenters = initialize_segmenters(corpus, debug=False)                 # Step 2: Sentence Segmentation
    _, tokens = preprocess_data(segmenters=segmenters, generate_new=False)  # Step 3: Preprocessing (Tokenization)
    train_word_embeddings()                                                 # Step 4: Developing Word Embeddings
    train_classifiers()                                                     # Step 5: Training Classifiers
    analyze_errors()                                                        # Step 6: Error Analysis


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Pass the path to the file containing the annotations and " \
                               "the directory with the unlabeled data => " \
                               "$ python pipeline.py " \
                               "<path to annotations file> <path to directory containing unlabeled data>"

    # Run train
    train(annotations_filepath=sys.argv[1], unlabeled_data_dir=sys.argv[2])

