"""
 File:   tokenizer.py
 Author: Batuhan Erden
"""

import random
import datetime
import json
import codecs
from tqdm import tqdm

from src.utils.type_utils import is_number, is_date
from src.utils.logging_utils import log


class Tokenizer:

    SENTENCE_SEGMENTED_DECISIONS_FILEPATH = "./data/unlabeled/_sentence_segmented_decisions.json"
    GENERATED_TOKENS_FILEPATH = "./data/unlabeled/_generated_tokens.json"

    MIN_NUM_TOKENS_IN_SENTENCE = 5

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
            sum([len(sentence_tokens) for sentence_tokens in sentences_with_tokens])
            for sentences_with_tokens in tokens_by_document.values()
        ])

    def sentence_segment_decisions_unlabeled(self):
        """
        Sentence-segments all decisions in the unlabeled corpus using a law-specific segmenter (Luima)

        :return: Sentence-segmented decisions
        """

        log("Sentence-segmenting all decisions in the unlabeled corpus using a law-specific segmenter (Luima)..")

        # Sentence-segment all decisions in the unlabeled corpus using a law-specific segmenter (Luima)
        sentences_by_document = self.segmenters["LuimaLawSegmenter"].apply_segmentation(annotated=False, debug=False)

        # Write generated sentences to file
        with open(self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH, "w") as file:
            json.dump(sentences_by_document, file)

        log("Sentence-segmented all decisions in the unlabeled corpus (%d sentences) and wrote them to %s!" %
            (Tokenizer.count_sentences(sentences_by_document), self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH))
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

            if is_date(str(token)):  # Simplify the date
                worthy_tokens.append(f"<DATE{len(token)}>")
            elif token.pos_ == "NUM" or is_number(str(token)):  # Simplify the number
                worthy_tokens.append(f"<NUM{len(token)}>")
            else:  # Otherwise, process the lemma
                # Lowercase lemma
                lemma = token.lemma_.lower()

                # Remove non-alphanumeric characters
                lemma = ''.join(char for char in lemma if char.isalnum())

                # Add if the resulting lemma is not empty
                if len(lemma) > 0:
                    worthy_tokens.append(lemma.lower())

        return worthy_tokens

    def generate_tokens_unlabeled(self, sentences_by_document):
        """
        Generates tokens from the sentence-segmented decisions in the unlabeled corpus using Spacy

        :param sentences_by_document: Sentence-segmented decisions
        :return: Tokens generated from the sentence-segmented decisions
        """

        log("Generating tokens from %d sentences in the unlabeled corpus!" %
            Tokenizer.count_sentences(sentences_by_document))

        spacy_segmenter = self.segmenters["ImprovedSpacySegmenter"]  # Improved Spacy segmenter
        spacy_segmenter.nlp.disable_pipes("parser")  # For a faster runtime

        # Start the timer
        start = datetime.datetime.now().replace(microsecond=0)

        # Generate tokens
        tokens_by_document = {
            document_id: [Tokenizer.tokenize(spacy_segmenter, sentence["txt"]) for sentence in sentences]
            for document_id, sentences in tqdm(sentences_by_document.items())
        }

        # End the timer and compute duration
        duration = datetime.datetime.now().replace(microsecond=0) - start

        # Write generated tokens to file
        with open(self.GENERATED_TOKENS_FILEPATH, "w") as file:
            json.dump(tokens_by_document, file)

        log("Generated %d tokens from the sentences in the unlabeled corpus and wrote them to %s! (Took %s.)" %
            (Tokenizer.count_tokens(tokens_by_document), duration, self.GENERATED_TOKENS_FILEPATH))
        return tokens_by_document

    def generate_unlabeled(self):
        """
        Generates the sentences and tokens in the unlabeled corpus

        Note: Takes a while... Get a coffee or take a nap - a very long one!

        :return: A tuple containing the sentences and tokens generated
        """

        sentences_by_document = self.sentence_segment_decisions_unlabeled()
        tokens_by_document = self.generate_tokens_unlabeled(sentences_by_document)

        return sentences_by_document, tokens_by_document

    def load_sentences_unlabeled(self):
        """
        Loads the existing sentences generated from the unlabeled corpus

        :return: The existing sentences
        """

        log("Loading the existing sentences generated from the unlabeled corpus..")
        sentences_by_document = json.load(open(self.SENTENCE_SEGMENTED_DECISIONS_FILEPATH))
        log("Loaded %d sentences generated from the unlabeled corpus!" %
            Tokenizer.count_sentences(sentences_by_document))

        return sentences_by_document

    def load_tokens_unlabeled(self):
        """
        Loads the existing tokens generated sentence-segmented decisions in the unlabeled corpus

        :return: The existing tokens
        """

        log("Loading the existing tokens generated from the sentence-segmented decisions in the unlabeled corpus..")
        tokens_by_document = json.load(open(self.GENERATED_TOKENS_FILEPATH))
        log("Loaded %d tokens generated from the sentence-segmented decisions in the unlabeled corpus!" %
            Tokenizer.count_tokens(tokens_by_document))

        return tokens_by_document

    def load_unlabeled(self):
        """
        Loads the existing sentences and tokens in the unlabeled corpus

        :return: A tuple containing the existing sentences and tokens loaded
        """

        return self.load_sentences_unlabeled(), self.load_tokens_unlabeled()

    def write_tokens_to_file_for_embeddings(self, sentences_by_document, tokens_by_document, filepath,
                                            randomized=True, random_seed=42):
        """
        Writes tokens to a file to be used to generate the word embeddings

        Each line of the file should consist of a sentence's tokens, separated by a single whitespace

        :param sentences_by_document: Sentence-segmented decisions
        :param tokens_by_document: Tokens generated
        :param filepath: The path of the file to which the tokens are written
        :param randomized: Whether or not the sentences are randomized (default: True)
        :param random_seed: Random seed used to randomize the inputs (default: 42)
        """

        # Flatten the sentences
        log("Flattening the sentences..")

        sentences_with_tokens_flattened = []
        for sentences_with_tokens in tokens_by_document.values():
            sentences_with_tokens_flattened += sentences_with_tokens

        # Randomize the sentences
        log("Randomizing the sentences..")

        if randomized:
            random.Random(random_seed).shuffle(sentences_with_tokens_flattened)

        # Write tokens to a string each line of which consists of a sentence's tokens, separated by a single whitespace
        log("Writing tokens to a file each line of which consists of a sentence's tokens, "
            "separated by a single whitespace..")

        tokens_text = ""
        for tokens in tqdm(sentences_with_tokens_flattened):
            if len(tokens) >= self.MIN_NUM_TOKENS_IN_SENTENCE:  # Only take into account the long enough tokens!
                tokens_text += " ".join(tokens) + "\n"

        # Write the collected string to file
        with codecs.open(filepath, "w", encoding="utf-8") as file:
            file.write(tokens_text)

        log("Tokens generated from %d of %d sentences successfully written to %s!" %
            (len(tokens_text.split("\n")), Tokenizer.count_sentences(sentences_by_document), filepath))

