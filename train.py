"""
 File:   train.py
 Author: Batuhan Erden
"""

import sys

from src.corpus import Corpus
from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter
from src.tokenizer import Tokenizer
from src.embeddings import Embeddings
from src.featurization.tfidf_featurizer import TfidfFeaturizer
from src.featurization.embeddings_featurizer import EmbeddingsFeaturizer
from src.utils.sys_utils import create_dir
from src.utils.logging_utils import log
from src.utils.classification_utils import train_using_featurizers

# Constants
OUT_DIR = "./out/"
GENERATED_TOKENS_FOR_EMBEDDINGS_FILEPATH = OUT_DIR + "_generated_tokens_for_embeddings.txt"
EMBEDDINGS_MODEL_FILEPATH = OUT_DIR + "_embeddings_model.bin"


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
    :return: Segmenters
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

    :param segmenters: Sentence segmenters
    :param generate_new: When set to True, sentences and tokens are generated from scratch. Otherwise, they are loaded
    """

    # Initialize the tokenizer
    tokenizer = Tokenizer(segmenters)

    # Generate or load sentences and tokens
    if generate_new:
        sentences_by_document, tokens_by_document = tokenizer.generate_unlabeled()  # Takes about 2 to 8 hours..
    else:
        sentences_by_document, tokens_by_document = tokenizer.load_unlabeled()

    # Write the tokens to a file to be used as an input to embedding computations
    tokenizer.write_tokens_to_file_for_embeddings(sentences_by_document, tokens_by_document,
                                                  filepath=GENERATED_TOKENS_FOR_EMBEDDINGS_FILEPATH)


def train_word_embeddings(train_new=False):
    """
    Step 4: Developing Word Embeddings

    :param train_new: When set to True, a new model is trained. Otherwise, the existing one is loaded (default: False)
    :return The embeddings model trained on the unlabeled corpus
    """

    # Initialize an embeddings model
    embeddings = Embeddings(model_filepath=EMBEDDINGS_MODEL_FILEPATH)

    # Train the model
    if train_new:
        embeddings.train(tokens_filepath=GENERATED_TOKENS_FOR_EMBEDDINGS_FILEPATH)

    return embeddings.model


def train_classifiers(corpus, segmenters, embeddings_model, debug=False):
    """
    Step 5: Training Classifiers

    :param corpus: Corpus
    :param segmenters: Sentence segmenters
    :param embeddings_model: The embeddings model trained on the unlabeled corpus
    :param debug: Whether or not the classifiers are tested (default: False)
    :return Linear and non-linear classifiers
    """

    # Step 5.1: TFIDF Featurization
    tfidf_featurizer = TfidfFeaturizer(corpus, tokenization_segmenter=segmenters["ImprovedSpacySegmenter"])
    tfidf_featurizer.vectorize()

    # Step 5.2: Word Embedding Featurization
    embeddings_featurizer = EmbeddingsFeaturizer(corpus, tokenization_segmenter=segmenters["ImprovedSpacySegmenter"],
                                                 embeddings_model=embeddings_model)

    # Step 5.3: Model Training
    return train_using_featurizers(featurizers={
        # "TfidfFeaturizer": tfidf_featurizer,  # Uncomment to also use TfidfFeaturizer
        "EmbeddingsFeaturizer": embeddings_featurizer
    }, debug=debug)


def analyze_errors(classifiers):
    """
    Step 6: Error Analysis

    :param classifiers: Linear and non-linear classifiers trained
    """

    log(f"Trained classifiers: {str([type(classifier).__name__ for classifier in classifiers])}")
    log("Please refer to 'train.ipynb' notebook for the detailed error analysis!")


def train(annotations_filepath, unlabeled_data_dir):
    """
    Trains the best model possible for BVA decisions

    :param annotations_filepath: Path to the file containing the annotations
    :param unlabeled_data_dir: Directory containing the unlabeled data
    """

    corpus = initialize_corpus(annotations_filepath, unlabeled_data_dir)    # Step 1: Dataset Splitting
    segmenters = initialize_segmenters(corpus, debug=False)                 # Step 2: Sentence Segmentation
    preprocess_data(segmenters=segmenters, generate_new=False)              # Step 3: Preprocessing (Tokenization)
    embeddings_model = train_word_embeddings(train_new=False)               # Step 4: Developing Word Embeddings
    classifiers = train_classifiers(corpus, segmenters, embeddings_model)   # Step 5: Training Classifiers
    analyze_errors(classifiers)                                             # Step 6: Error Analysis


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Pass the path to the file containing the annotations and " \
                               "the directory containing the unlabeled data => " \
                               "$ python train.py " \
                               "<path to annotations file> <path to directory containing unlabeled data>"

    # Create the out directory
    create_dir(OUT_DIR)

    # Run train
    train(annotations_filepath=sys.argv[1], unlabeled_data_dir=sys.argv[2])

