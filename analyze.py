"""
 File:   analyze.py
 Author: Batuhan Erden
"""

import sys
import pickle
from texttable import Texttable

from src.corpus import Corpus
from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter
from src.embeddings import Embeddings
from src.featurization.embeddings_featurizer import EmbeddingsFeaturizer
from src.utils.sys_utils import create_dir
from src.utils.logging_utils import log

# Constants
OUT_DIR = "./out/"
EMBEDDINGS_MODEL_FILEPATH = OUT_DIR + "_embeddings_model.bin"
BEST_CLASSIFIER_FILEPATH = OUT_DIR + "_best_classifier.pkl"


def log_results(sentences, predicted_labels):
    """
    Logs the results in a beautiful table

    :param sentences: Sentences split
    :param predicted_labels: Predicted labels for the sentences
    """
    table = Texttable()

    table.set_cols_valign(["m", "m"])
    table.add_rows(
        [["Sentence", "Predicted Label"]] +
        [[sentence["txt"], predicted_labels[idx]] for idx, sentence in enumerate(sentences)]
    )

    log(f"The resulting splits and the predicted labels for them:\n{table.draw()}")


def analyze(bva_decision_filepath):
    """
    Makes a prediction using the given BVA decision

    :param bva_decision_filepath: BVA decision to be predicted
    """

    log(f"Loading the BVA decision from {bva_decision_filepath}..")

    # Load the BVA decision
    with open(bva_decision_filepath, encoding="latin-1") as data:
        bva_decision_plain_text = data.read()

    log(f"The BVA decision successfully loaded from {bva_decision_filepath}!")
    log("Sentence-segmenting the BVA decision loaded..")

    # Sentence-segment BVA decision using Luima segmenter
    sentences = LuimaLawSegmenter(corpus=None).generate_sentences(bva_decision_plain_text)

    # Create span data from sentences generated and add it to the Corpus
    spans = [
        Corpus.create_span(
            plainText=bva_decision_plain_text, txt=sentence["txt"],
            start=sentence["start_char"], end=sentence["end_char"])
        for sentence in sentences
    ]

    log("The BVA decision successfully split into sentences!")

    # Load the embeddings model and initialize word embedding featurization
    embeddings = Embeddings(model_filepath=EMBEDDINGS_MODEL_FILEPATH)
    embeddings_featurizer = EmbeddingsFeaturizer(corpus=None,
                                                 tokenization_segmenter=SpacySegmenter(corpus=None, improved=True),
                                                 embeddings_model=embeddings.model)

    log("Creating the inputs to be fed into the network..")

    # Create the inputs from the sentence-segmented BVA decision
    X, _ = embeddings_featurizer.create_inputs_and_labels_for_spans(dataset_name="analyzed",
                                                                    spans=spans, tokenize=True)

    log("The inputs to be fed into the network successfully created!")
    log(f"Loading the best classifier from {BEST_CLASSIFIER_FILEPATH}..")

    # Load the best classifier saved
    classifier = pickle.load(open(BEST_CLASSIFIER_FILEPATH, "rb"))

    log(f"The best classifier successfully loaded from {BEST_CLASSIFIER_FILEPATH}!")
    log("Classifying the given BVA decision..")

    # Make prediction and log the results
    predicted_labels = classifier.predict(X)

    # Log the results
    log_results(sentences, predicted_labels)

    log("The given BVA decision successfully classified!")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Pass the path to the text file containing a BVA decision => " \
                               "$ python analyze.py <path to txt file>"

    # Create the out directory
    create_dir(OUT_DIR)

    # Run analyze
    analyze(bva_decision_filepath=sys.argv[1])

