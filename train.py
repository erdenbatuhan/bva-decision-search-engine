"""
 File:   train.py
 Author: Batuhan Erden
"""

import sys

from src.corpus import Corpus

from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter

from src.tokenizer import Tokenizer


OUT_DIRECTORY = "./out/"


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


def preprocess_data(segmenter=None, use_pre_split_sentences=True):
    """
    Step 3: Preprocessing (Tokenization)

    :param segmenter: Segmenter used (Will not be used if use_pre_split_sentences is set to True)
    :param use_pre_split_sentences: Whether the pre-split sentences are used or new ones are generated
    """

    # TODO: Read pre-split sentences
    if use_pre_split_sentences:
        # pre_split_sentence_dir = OUT_DIRECTORY + "unlabeled.txt"
        sentences = None
    else:
        sentences = segmenter.apply_segmentation(annotated=False, debug=False).values()

    # Initialize the tokenizer
    tokenizer = Tokenizer(sentences)


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

    corpus = initialize_corpus(annotations_filepath, unlabeled_data_dir)  # Step 1: Dataset Splitting
    segmenters = initialize_segmenters(corpus)                            # Step 2: Sentence Segmentation
    preprocess_data(segmenter=segmenters["LuimaLawSegmenter"])            # Step 3: Preprocessing (Tokenization)
    train_word_embeddings()                                               # Step 4: Developing Word Embeddings
    train_classifiers()                                                   # Step 5: Training Classifiers
    analyze_errors()                                                      # Step 6: Error Analysis


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Pass the path to the file containing the annotations and " \
                               "the directory with the unlabeled data => " \
                               "$ python train.py " \
                               "<path to annotations file> <path to directory containing unlabeled data>"

    # Run train
    train(annotations_filepath=sys.argv[1], unlabeled_data_dir=sys.argv[2])

