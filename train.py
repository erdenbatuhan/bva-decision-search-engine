"""
 File:   train.py
 Author: Batuhan Erden
"""

import sys

from src.corpus import Corpus
from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter


def train(corpus_fpath):
    """
    Analyzes multiple methods to achieve the best performance

    :param corpus_fpath: Path to the file containing the annotations
    """

    """
    ====================================
    Step 1: Initialize Corpus & Dataset Splitting
    ====================================
    """
    corpus = Corpus(corpus_fpath=corpus_fpath)
    print(corpus)

    """
    ====================================
    Step 2: Sentence Segmentation
    ====================================
    """
    # Step 2.1: Standard segmentation analysis
    # naive_spacy_segmenter = SpacySegmenter(corpus=corpus)
    # naive_spacy_segmenter.apply_segmentation()

    # Step 2.2: Improved segmentation analysis
    # improved_spacy_segmenter = SpacySegmenter(corpus=corpus, improved=True)
    # improved_spacy_segmenter.apply_segmentation()

    # Step 2.2: Improved segmentation analysis
    luima_law_segmenter = LuimaLawSegmenter(corpus=corpus)
    luima_law_segmenter.apply_segmentation()

    """
    ====================================
    Step 3: Preprocessing
    ====================================
    """
    pass

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
    assert len(sys.argv) == 2, "Pass the path to the file containing the annotations => " \
                               "$ python train.py <path to annotations file>"

    # Run train
    train(corpus_fpath=sys.argv[1])

