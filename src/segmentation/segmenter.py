"""
 File:   segmenter.py
 Author: Batuhan Erden
"""

import numpy as np


class Segmenter:

    MAX_MATCHING_DIST = 3

    def __init__(self, corpus):
        self.corpus = corpus

    @staticmethod
    def get_true_splits_by_document(spans):
        """
        Gets true splits by document given spans

        :param spans: Spans given
        :return: True splits by document given spans
        """

        true_splits_by_document = {}

        for span in spans:
            document_id = span["document"]

            if document_id not in true_splits_by_document:
                true_splits_by_document[document_id] = []

            true_splits_by_document[document_id].append((span["start"], span["end"]))

        return true_splits_by_document

    def compare_splits(self, true_splits, generated_splits):
        """
        Compares splits

        For each true and generated split:
            start_diff = abs(true_start - generated_start)
            end_diff = abs(true_end - generated_end)

            Match found if both are smaller than MAX_MATCHING_DIST!

        :param true_splits: True splits from Corpus
        :param generated_splits: Generated splits from generates sentences by Spacy
        :return: A tuple containing the scores (true positives, false negatives, false positives)
        """

        true_positives, false_negatives, false_positives = 0, 0, 0

        # Compare true and generated splits
        for i in range(min(len(true_splits), len(generated_splits))):
            start_diff, end_diff = abs(np.array(true_splits[i]) - np.array(generated_splits[i]))
            match_found = start_diff < self.MAX_MATCHING_DIST and end_diff < self.MAX_MATCHING_DIST

            true_positives += match_found
            false_negatives += not match_found
            false_positives += not match_found

        # Add the number of unmatched to false negatives or false positives depending on the bigger list
        false_negatives += max(0, len(true_splits) - len(generated_splits))
        false_positives += max(0, len(generated_splits) - len(true_splits))

        return true_positives, false_negatives, false_positives

    def calculate_split_scores(self, generated_sentences_by_document):
        """
        Compares the generated splits with true splits and calculates the scores

        - True Positive when true and generated splits are "matched" being within MAX_MATCHING_DIST chars of each other
        - Unmatched true splits are false negatives
        - Unmatched generated splits are false positives

        :param generated_sentences_by_document: Sentences generated by Spacy
        :return: A tuple containing the scores (TPs, FNs, FPs, precision, recall, f1_score)
        """

        true_positives, false_negatives, false_positives = 0, 0, 0

        # Get the true splits (starts and ends) from corpus
        true_splits_by_document = self.get_true_splits_by_document(self.corpus.train_spans)

        # Iterate over documents
        for document_id, true_splits in true_splits_by_document.items():
            # Get the generated splits (starts and ends) from generated sentences
            generated_splits = [
                (sentence.start_char, sentence.end_char) for sentence in generated_sentences_by_document[document_id]
            ]

            # Compare splits for current document
            local_true_positives, local_false_negatives, local_false_positives = \
                self.compare_splits(true_splits, generated_splits)

            # Add local scores to global scores
            true_positives += local_true_positives
            false_negatives += local_false_negatives
            false_positives += local_false_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * precision * recall / (precision + recall)

        return true_positives, false_negatives, false_positives, precision, recall, f1_score

    def analyze_segmentation(self, name, generated_sentences_by_document):
        """
        Analyzes segmentation (Error analysis)

        :param name: Name of the segmenter
        :param generated_sentences_by_document: Generated sentences by document
        """
        split_scores = self.calculate_split_scores(generated_sentences_by_document)

        print("===========================================================================\n" +
              ("Segmentation Error Analysis for %s:\n" % name) +
              "===========================================================================\n" +
              ("True Positives: %d\nFalse Negatives: %d\nFalse Positives: %d\n" % split_scores[:3]) +
              ("Precision: %f\nRecall: %f\nF1 Score: %f\n" % split_scores[3:]) +
              "===========================================================================\n")

