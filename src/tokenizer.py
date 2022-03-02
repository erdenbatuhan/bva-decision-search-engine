import json

from src.utils.logging_utils import log


class Tokenizer:

    SENTENCE_SEGMENTED_DECISIONS_FILEPATH = "./data/sentence_segmented_decisions.json"

    def __init__(self, segmenter):
        self.sentences = self.load_sentence_segmented_decisions(segmenter)

    def load_sentence_segmented_decisions(self, segmenter, use_existing=True):
        """
        Loads the sentence-segmented decisions of the unlabeled corpus

        :param use_existing: Whether or not the existing sentence-segmented decisions are used
        :param segmenter: Segmenter used to sentence-segment the decisions in the unlabeled corpus
        :return: Sentence-segmented decisions
        """

        # Load the existing sentence-segmented decisions from file
        if use_existing:
            try:
                return json.load(open(self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH))
            except FileNotFoundError:
                log("No sentence-segmented decision is found! Generating new ones.. (This may take a while ~2h)")

        # Sentence-segment all decisions in the unlabeled corpus
        sentences = segmenter.apply_segmentation(annotated=False, debug=False)

        # Write generated sentences to file
        with open(self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH, "w") as file:
            json.dump(sentences, file)

        # Return sentences
        return sentences

