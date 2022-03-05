"""
 File:   analyze.py
 Author: Batuhan Erden
"""

import sys
import numpy as np
import pandas as pd
import pickle

from src.corpus import Corpus
from src.segmentation.spacy_segmenter import SpacySegmenter
from src.segmentation.luima_law_segmenter import LuimaLawSegmenter
from src.embeddings import Embeddings
from src.featurization.embeddings_featurizer import EmbeddingsFeaturizer
from src.utils.logging_utils import log

# Constants
OUT_DIR = "./out/"
GENERATED_TOKENS_FOR_EMBEDDINGS_FILEPATH = OUT_DIR + "_generated_tokens_for_embeddings.txt"
EMBEDDINGS_MODEL_FILEPATH = OUT_DIR + "_embeddings_model.bin"
BEST_CLASSIFIER_FILEPATH = OUT_DIR + "_best_classifier.pkl"


def analyze(bva_decision_filepath):
    """
    Analyzes the performance of the model
    """

    # Load the BVA decision
    with open(bva_decision_filepath, encoding="latin-1") as data:
        bva_decision_plain_text = data.read()

    # Sentence-segment BVA decision using Luima segmenter
    sentences = LuimaLawSegmenter(corpus=None).generate_sentences(bva_decision_plain_text)

    # Create span data from sentences generated and add it to the Corpus
    spans = [
        Corpus.create_span(
            plainText=bva_decision_plain_text, txt=sentence["txt"],
            start=sentence["start_char"], end=sentence["end_char"])
        for sentence in sentences
    ]

    # Load the embeddings model and initialize word embedding featurization
    embeddings = Embeddings(model_filepath=EMBEDDINGS_MODEL_FILEPATH)
    embeddings_featurizer = EmbeddingsFeaturizer(corpus=None,
                                                 tokenization_segmenter=SpacySegmenter(corpus=None, improved=True),
                                                 embeddings_model=embeddings.model)

    # Create the inputs from the sentence-segmented BVA decision
    X, _ = embeddings_featurizer.create_inputs_and_labels_for_spans(dataset_name="analyzed",
                                                                    spans=spans, tokenize=True)

    # Load the best classifier saved and make prediction
    classifier = pickle.load(open(BEST_CLASSIFIER_FILEPATH, "rb"))
    predicted_labels = classifier.predict(X)

    # Log the results
    results_df = pd.DataFrame(data={
        "Sentence": [sentence["txt"] for sentence in sentences],
        "Predicted Label": predicted_labels
    })

    # Set pandas display options
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', int(np.mean([len(sentence["txt"]) for sentence in sentences])))
    pd.set_option("display.max_rows", results_df.shape[0] + 1)

    log(f"The resulting splits and the predicted labels for them:\n{results_df}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Pass the path to the text file containing a BVA decision => " \
                               "$ python analyze.py <path to txt file>"

    # Run analyze
    analyze(bva_decision_filepath=sys.argv[1])

