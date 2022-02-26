"""
 File:   sentence_segmenter.py
 Author: Batuhan Erden
"""

import numpy as np
import spacy


class SentenceSegmenter:

    MAX_MATCHING_DIST = 3

    def __init__(self, corpus):
        self.corpus = corpus

    def calculate_split_scores(self, generated_sentences_by_document):
        """
        Compares the generated splits with true splits and calculates the scores

        :param generated_sentences_by_document: Sentences generated by Spacy
        :return: A tuple containing the scores (true positives, false negatives, false positives)
        """

        true_positives = 0  # True and generated splits are "matched" if they are within 3 characters of each other
        false_negatives = 0  # Unmatched true splits are false negatives
        false_positives = 0  # Unmatched generated splits are false positives

        # Get the true splits from corpus
        true_splits_by_document = self.corpus.get_true_splits_by_document(self.corpus.train_spans)

        for document_id, true_splits in true_splits_by_document.items():
            generated_splits = [
                (sentence.start, sentence.end) for sentence in generated_sentences_by_document[document_id]
            ]

            for i in range(min(len(true_splits), len(generated_splits))):
                start_diff, end_diff = np.array(true_splits[i]) - np.array(generated_splits[i])
                match_found = abs(end_diff - start_diff) < self.MAX_MATCHING_DIST

                true_positives += match_found
                false_negatives += not match_found
                false_positives += not match_found

            false_negatives += max(0, len(true_splits) - len(generated_splits))
            false_positives += max(0, len(generated_splits) - len(true_splits))

        return true_positives, false_negatives, false_positives

    def spacy_segmentation(self, nlp, error_analysis=False):
        """
        Generates sentences using Spacy

        :param nlp: Spacy's NLP instance
        :param error_analysis: Error analysis will be conducted when True
        :return: Sentences generated by Spacy
        """

        # Generate spacy sentences
        generated_sentences_by_document = {
            document_id: list(nlp(self.corpus.annotated_documents_by_id[document_id]["plainText"]).sents)
            for document_id in self.corpus.get_documents_split(self.corpus.train_spans)
        }

        if error_analysis:  # Do error analysis
            split_scores = self.calculate_split_scores(generated_sentences_by_document)
            print(split_scores)

        return generated_sentences_by_document

    def spacy_segmentation_naive(self, error_analysis=False):
        """
        Generates sentences using Naive Spacy "without" additional exceptions/extensions

        :param error_analysis: Error analysis will be conducted when True
        :return: Sentences generated by Naive Spacy "without" additional exceptions/extensions
        """

        # Load basic English pipeline provided by spacy
        nlp = spacy.load("en_core_web_sm")

        return self.spacy_segmentation(nlp, error_analysis)

    def spacy_segmentation_improved(self, error_analysis=False):
        """
        Generates sentences using Improved Spacy "with" additional exceptions/extensions

        :param error_analysis: Error analysis will be conducted when True
        :return: Sentences generated by Improved Spacy "with" additional exceptions/extensions
        """

        # Load basic English pipeline provided by spacy
        nlp = spacy.load("en_core_web_sm")

        # TODO: Add some constraints here

        return self.spacy_segmentation(nlp, error_analysis)

