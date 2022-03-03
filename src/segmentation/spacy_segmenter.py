"""
 File:   spacy_segmenter.py
 Author: Batuhan Erden
"""

import spacy

from src.segmentation.segmenter import Segmenter


class SpacySegmenter(Segmenter):

    def __init__(self, corpus, improved=False):
        super().__init__(name="Spacy Segmentation (%s)" % ("Naive" if not improved else "Improved"),
                         corpus=corpus)

        # Load basic English pipeline provided by spacy
        self.nlp = spacy.load("en_core_web_sm")

        if improved:  # Introduce additional exceptions/extensions
            self.extend_language()

    def extend_language(self):
        """
        Introduces additional exceptions/extensions to the language/segmenter

        (1) Handles special legal words:
            "Vet. App.", "Fed. Cir.", "Fed. Reg.", "Pub. L. No.", "DOCKET NO.", "), DATE))", "non-Federal", "CF. 38"
        (2) Handles commas and semicolons after closed parenthesis: (2004), and (2004);
        (3) Handles the footer underscores
        """

        # Handle special legal words (1) and commas and semicolons after closed parenthesis (2)
        for word in [
            "Vet. App.", "Fed. Cir.", "Fed. Reg.", "Pub. L. No.", "DOCKET NO.", ")DATE))", "non-Federal", "Cf. 38"
        ] + ["), ", "); "] + ["____________________________________________"]:
            self.nlp.tokenizer.add_special_case(word, [{"ORTH": word}])

    def generate_sentences(self, plain_text):
        """
        Generates sentences from a plain text using Spacy

        Overrides the definition in the super class

        :param plain_text: The plain text
        :return: Sentences generated from a plain text by Spacy
        """

        return [
            {
                "text": sentence.text,
                "start_char": sentence.start_char,
                "end_char": sentence.end_char,
            }
            for sentence in self.nlp(plain_text).sents
        ]

