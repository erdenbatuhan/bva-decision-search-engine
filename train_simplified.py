"""
 File:   train_simplified.py
 Author: Batuhan Erden

 This script is just a shorter version of "train_detailed.ipynb",
 please see "train_detailed.ipynb" notebook for more detailed pipeline.
"""

import sys
import numpy as np
import pickle

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.corpus import Corpus
from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter
from src.tokenizer import Tokenizer
from src.embeddings import Embeddings
from src.featurization.tfidf_featurizer import TfidfFeaturizer
from src.featurization.embeddings_featurizer import EmbeddingsFeaturizer
from src.utils.sys_utils import create_dir
from src.utils.logging_utils import log
from src.utils.classification_utils import test_classifier

# Constants
OUT_DIR = "./out/"
GENERATED_TOKENS_FOR_EMBEDDINGS_FILEPATH = OUT_DIR + "_generated_tokens_for_embeddings.txt"
EMBEDDINGS_MODEL_FILEPATH = OUT_DIR + "_embeddings_model.bin"
BEST_CLASSIFIER_FILEPATH = OUT_DIR + "_best_classifier.pkl"


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
    tokenizer = Tokenizer(sentence_segmenter=segmenters["LuimaLawSegmenter"],
                          tokenization_segmenter=segmenters["ImprovedSpacySegmenter"])

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

    # Log the results
    log("Vocabulary size: %d words and %d labels" %
        (len(embeddings.model.words), len(embeddings.model.labels)))
    log("Shape of the embeddings model: (%d, %d)" %
        (np.shape(embeddings.model.words)[0], embeddings.model.get_dimension()))

    return embeddings.model


def train_best_classifier(corpus, segmenters, embeddings_model):
    """
    Step 5: Training Classifier(s)

    Training of different classifiers: Detailed tests for different featurizers with different classifiers
                                       can be found in the "train_detailed.ipynb" notebook

    :param corpus: Corpus
    :param segmenters: Sentence segmenters
    :param embeddings_model: The embeddings model trained on the unlabeled corpus
    """

    log("Training and saving the best classifier..")

    # Word Embedding Featurization (See the "train_detailed.ipynb" notebook for how to use the TFIDF feature).
    embeddings_featurizer = EmbeddingsFeaturizer(corpus, tokenization_segmenter=segmenters["ImprovedSpacySegmenter"],
                                                 embeddings_model=embeddings_model)

    # Create the inputs and labels
    X, y = embeddings_featurizer.create_inputs_and_labels()
    log(embeddings_featurizer.analyze_shapes(X, y))

    # Train the classifier
    classifier = RandomForestClassifier(n_estimators=100, max_depth=15, bootstrap=True)
    classifier.fit(X["train"], y["train"])

    # Save the classifier
    pickle.dump(classifier, open(BEST_CLASSIFIER_FILEPATH, "wb"))

    log("The best classifier successfully trained and saved!")


def train(annotations_filepath, unlabeled_data_dir):
    """
    Trains the best classifier possible for BVA decisions

    :param annotations_filepath: Path to the file containing the annotations
    :param unlabeled_data_dir: Directory containing the unlabeled data
    """

    corpus = initialize_corpus(annotations_filepath, unlabeled_data_dir)    # Step 1: Dataset Splitting
    segmenters = initialize_segmenters(corpus, debug=False)                 # Step 2: Sentence Segmentation
    preprocess_data(segmenters=segmenters, generate_new=False)              # Step 3: Preprocessing (Tokenization)
    embeddings_model = train_word_embeddings(train_new=False)               # Step 4: Developing Word Embeddings
    train_best_classifier(corpus, segmenters, embeddings_model)             # Step 5: Training Classifiers

    # Step 6: Error Analysis
    log("Please refer to 'train_detailed.ipynb' notebook for the detailed error analysis!")


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Pass the path to the file containing the annotations and " \
                               "the directory containing the unlabeled data => " \
                               "$ python train_simplified.py " \
                               "<path to annotations file> <path to directory containing unlabeled data>"

    # Create the out directory
    create_dir(OUT_DIR)

    # Run train
    train(annotations_filepath=sys.argv[1], unlabeled_data_dir=sys.argv[2])

