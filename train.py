"""
 File:   train.py
 Author: Batuhan Erden
"""

import sys

from src.corpus import Corpus

from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter

from src.tokenizer import Tokenizer


def train(annotations_filepath, unlabeled_data_dir):
    """
    Analyzes multiple methods to achieve the best performance

    :param annotations_filepath: Path to the file containing the annotations
    :param unlabeled_data_dir: Directory containing the unlabeled data
    """

    """
    ====================================
    Step 1: Initialize the Corpus & Dataset Splitting
    ====================================
    """
    # Initialize the Corpus
    corpus = Corpus(annotations_filepath, unlabeled_data_dir)
    print(corpus)

    """
    ====================================
    Step 2: Sentence Segmentation
    ====================================
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
    for segmenter in segmenters.values():
        segmenter.apply_segmentation()

    """
    ====================================
    Step 3: Preprocessing
    ====================================
    """
    # Initialize the tokenizer with Luima segmenter
    tokenizer = Tokenizer(segmenter=segmenters["LuimaLawSegmenter"])

    """
    ====================================
    Step 4: Developing Word Embeddings
    ====================================
    """
    pass

    """
    ====================================
    Step 5: Training Classifiers
    ====================================
    """
    pass

    """
    ====================================
    Step 6: Error Analysis
    ====================================
    """
    pass


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Pass the path to the file containing the annotations and " \
                               "the directory with the unlabeled data => " \
                               "$ python train.py " \
                               "<path to annotations file> <path to directory containing unlabeled data>"

    # Run train
    train(annotations_filepath=sys.argv[1], unlabeled_data_dir=sys.argv[2])

