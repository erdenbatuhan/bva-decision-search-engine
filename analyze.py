"""
 File:   analyze.py
 Author: Batuhan Erden
"""

import sys

from src.models.corpus import Corpus
from src.models.sentence_segmenter import SentenceSegmenter


def analyze(corpus_fpath="./data/ldsi_w21_curated_annotations_v2.json"):
    """
    Analyzes multiple methods to achieve the best performance

    :param corpus_fpath: Path to the annotations file
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
    sentence_segmenter = SentenceSegmenter(corpus=corpus)

    # Step 2.1: Standard segmentation analysis
    sentence_segmenter.spacy_segmentation_naive(error_analysis=True)

    # Step 2.2: Improved segmentation analysis
    sentence_segmenter.spacy_segmentation_improved(error_analysis=True)

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
    assert len(sys.argv) == 2, "Pass the path to the text file containing a BVA decision => " \
                               "$ python analyze.py <path to txt file>"

    # Run analyze
    analyze()

