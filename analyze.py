import sys

from corpus import Corpus


def analyze(corpus_fpath="./data/ldsi_w21_curated_annotations_v2.json"):
    # Initialize Corpus (Phase 1 - Dataset Splitting)
    corpus = Corpus(corpus_fpath)
    print(corpus)

    # Sentence Segmentation (Phase 2)
    pass

    # Preprocessing (Phase 3)
    pass

    # Developing Word Embeddings (Phase 4)
    pass

    # Training Classifiers (Phase 5)
    pass

    # Error Analysis (Phase 6)
    pass


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Pass the path to the text file containing a BVA decision => " \
                               "$ python analyze.py <path to txt file>"

    # Run analyze with the filepath provided
    analyze()

