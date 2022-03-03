"""
 File:   unlabeled_tokenizer.py
 Author: Batuhan Erden
"""

import datetime
import json

from src.utils.logging_utils import log


class UnlabeledTokenizer:

    SENTENCE_SEGMENTED_DECISIONS_FILEPATH = "./data/unlabeled/_sentence_segmented_decisions.json"
    GENERATED_TOKENS_FILEPATH = "./data/unlabeled/_generated_tokens.json"

    def __init__(self, segmenters):
        self.segmenters = segmenters

    @staticmethod
    def count_sentences(sentences_by_document):
        """
        Counts the total number of sentences

        :param sentences_by_document: Sentence-segmented decisions
        :return: The total number of sentences
        """

        return sum([len(sentences) for sentences in sentences_by_document.values()])

    @staticmethod
    def count_tokens(tokens_by_document):
        """
        Counts the total number of tokens

        :param tokens_by_document: Tokens generated
        :return: The total number of tokens
        """

        return sum([
            sum([len(tokens) for tokens in sentence_tokens])
            for sentence_tokens in tokens_by_document.values()
        ])

    def sentence_segment_decisions(self):
        """
        Sentence-segments all decisions in the unlabeled corpus using a law-specific segmenter (Luima)

        :return: Sentence-segmented decisions
        """

        # Sentence-segment all decisions in the unlabeled corpus using a law-specific segmenter (Luima)
        sentences_by_document = self.segmenters["LuimaLawSegmenter"].apply_segmentation(annotated=False, debug=False)

        # Write generated sentences to file
        with open(self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH, "w") as file:
            json.dump(sentences_by_document, file)

        log("Generated %d sentences from the unlabeled corpus!" % self.count_sentences(sentences_by_document))
        return sentences_by_document

    @staticmethod
    def tokenize(spacy_segmenter, plain_text):
        """
        Tokenizes given sentence using Spacy and some additional extensions

        :param spacy_segmenter: Spacy segmenter
        :param plain_text: Sentence to be processed
        :return: Tokens generated from the given sentence
        """

        tokens = list(spacy_segmenter.nlp(plain_text))  # Generate tokens using Spacy
        worthy_tokens = []

        # Extract only the worthy tokens
        for token in tokens:
            # Remove punctuation, space or particle
            if token.pos_ in ["PUNCT", "SPACE", "PART"]:
                continue

            # Remove the special characters: "�" and "§"
            if token.lemma_ in ["�", "§"]:
                continue

            if token.pos_ == "NUM":  # Simplify the number
                worthy_tokens.append(f"<NUM{len(token)}>")
            else:  # Lowercase lemma
                worthy_tokens.append(token.lemma_.lower())

        return worthy_tokens

    def generate_tokens(self, sentences_by_document):
        """
        Generates tokens from the sentence-segmented decisions in the unlabeled corpus using Spacy

        :param sentences_by_document: Sentence-segmented decisions
        :return: Tokens generated from the sentence-segmented decisions
        """

        spacy_segmenter = self.segmenters["ImprovedSpacySegmenter"]  # Improved Spacy segmenter
        spacy_segmenter.nlp.disable_pipes("parser")  # For a faster runtime

        # Start the timer
        start = datetime.datetime.now().replace(microsecond=0)

        # Generate tokens
        tokens_by_document = {
            document_id: [self.tokenize(spacy_segmenter, sentence["text"]) for sentence in sentences]
            for document_id, sentences in sentences_by_document.items()
        }

        # End the timer and compute duration
        duration = datetime.datetime.now().replace(microsecond=0) - start

        # Write generated tokens to file
        with open(self.GENERATED_TOKENS_FILEPATH, "w") as file:
            json.dump(tokens_by_document, file)

        log("Generated %d tokens from the sentences in the unlabeled corpus! (Took %s.)" %
            (self.count_tokens(tokens_by_document), duration))

        return tokens_by_document

    def generate(self):
        """
        Generates the sentences and tokens (Takes a while... Get a coffee or take a nap - a very long one!)

        :return: A tuple containing the sentences and tokens generated
        """

        # sentences_by_document = self.sentence_segment_decisions()
        sentences_by_document = self.load_sentences()
        tokens_by_document = self.generate_tokens(sentences_by_document)

        return sentences_by_document, tokens_by_document

    def load_sentences(self):
        """
        Loads the existing sentences

        :return: The existing sentences
        """

        sentences_by_document = json.load(open(self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH))

        log("Loaded %d sentences generated from the unlabeled corpus!" % self.count_sentences(sentences_by_document))
        return sentences_by_document

    def load_tokens(self):
        """
        Loads the existing tokens

        :return: The existing tokens
        """

        tokens_by_document = json.load(open(self.GENERATED_TOKENS_FILEPATH))

        log("Loaded %d tokens generated from the unlabeled corpus!" % self.count_tokens(tokens_by_document))
        return tokens_by_document

    def load(self):
        """
        Loads the existing sentences and tokens

        :return: A tuple containing the existing sentences and tokens loaded
        """

        return self.load_sentences(), self.load_tokens()

