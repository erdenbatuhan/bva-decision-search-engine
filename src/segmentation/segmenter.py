"""
File:   segmenter.py
Author: Batuhan Erden
"""

import datetime
import numpy as np
from abc import abstractmethod
from tqdm import tqdm

from src.utils.logging_utils import log


class Segmenter:

    MAX_MATCHING_DIST = 3

    def __init__(self, name, corpus):
        self.name = name
        self.corpus = corpus

        # Metrics related to error analysis
        self.error_analysis_metrics = {
            "deep_compare": True  # With the setting "false", only the difference of the start indices is compared!
        }

    def get_plain_texts_by_document(self, annotated=True):
        """
        Returns the plain texts from annotated or unannotated documents

        :param annotated: Whether or not the data returned is annotated (default: True)
        :return: The plain texts from annotated or unannotated documents
        """
        if annotated:
            return self.corpus.get_plain_texts_by_annotated_document(self.corpus.train_spans)

        return self.corpus.unlabeled_data

    @abstractmethod
    def generate_sentences(self, plain_text):
        raise Exception("This is an abstract method that needs to be overridden!")

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

            Match found if both are smaller than or equal to MAX_MATCHING_DIST!

        :param true_splits: True splits from Corpus
        :param generated_splits: Generated splits from generates sentences by Spacy
        :return: A tuple containing the scores (true positives, false negatives, false positives)
        """
        true_positives = 0

        # Compare true and generated splits
        for true_split in true_splits:
            for generated_split in generated_splits:
                start_diff, end_diff = abs(np.array(true_split) - np.array(generated_split))

                # Disable the comparison of the difference of the end indices
                if not self.error_analysis_metrics["deep_compare"]:
                    end_diff = self.MAX_MATCHING_DIST

                if start_diff <= self.MAX_MATCHING_DIST and end_diff <= self.MAX_MATCHING_DIST:  # Match found!
                    true_positives += 1
                    break

        false_negatives, false_positives = len(true_splits) - true_positives, len(generated_splits) - true_positives
        return true_positives, false_negatives, false_positives

    @staticmethod
    def calculate_measurement_scores(true_positives, false_negatives, false_positives):
        """
        Calculates the measurement scores like precision, recall and f1

        :param true_positives: TPs
        :param false_negatives: FNs
        :param false_positives: FPs
        :return: A tuple containing the measurement scores (precision, recall, f1_score)
        """
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f1_score

    def calculate_split_scores(self, generated_sentences_by_document):
        """
        Compares the generated splits with true splits and calculates the scores

        - True Positive when true and generated splits are "matched" being within MAX_MATCHING_DIST chars of each other
        - Unmatched true splits are false negatives
        - Unmatched generated splits are false positives

        :param generated_sentences_by_document: Sentences generated by Spacy
        :return: An object containing the score for each document (TPs, FNs, FPs, precision, recall, f1_score)
        """
        scores_by_document = {}

        # Get the true splits (starts and ends) from corpus
        true_splits_by_document = Segmenter.get_true_splits_by_document(self.corpus.train_spans)

        # Iterate over documents
        for document_id, true_splits in tqdm(true_splits_by_document.items()):
            # Get the generated splits (starts and ends) from generated sentences
            generated_splits = [
                (sentence["start_char"], sentence["end_char"])
                for sentence in generated_sentences_by_document[document_id]
            ]

            # Compare splits for current document
            true_positives, false_negatives, false_positives = self.compare_splits(true_splits, generated_splits)

            # Calculate measured scores
            precision, recall, f1_score = Segmenter.calculate_measurement_scores(
                true_positives, false_negatives, false_positives)

            # Add scores to the dictionary
            scores_by_document[document_id] = {
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }

        return scores_by_document

    @staticmethod
    def get_worst_scores(split_scores_by_document, performance_deciding_attribute="f1_score", n=3):
        """
        Gets the "n" document(s) with the worst performance in terms of score

        :param split_scores_by_document: An object containing the score for each document (:see calculate_split_scores)
        :param performance_deciding_attribute: The attribute used to compare performance (default: f1_score)
        :param n: Number of documents listed (default: 3)
        :return: Split scores of "n" worst performing documents
        """
        # Define the function to be used in sorting
        get_performance_value = lambda document_id: \
            split_scores_by_document[document_id][performance_deciding_attribute]

        # Find the "n" document(s) with the worst performance in terms of score
        worst_document_ids = sorted(split_scores_by_document, key=get_performance_value)[:n]

        # Return the documents with their score values
        return {document_id: split_scores_by_document[document_id] for document_id in worst_document_ids}

    def analyze_segmentation(self, generated_sentences_by_document, duration):
        """
        Analyzes segmentation (Error analysis)

        :param generated_sentences_by_document: Generated sentences by document
        :param duration: Time it took to apply segmentation
        """
        split_scores_by_document = self.calculate_split_scores(generated_sentences_by_document)

        # Calculate total scores
        total_true_positives = sum([score["true_positives"] for score in split_scores_by_document.values()])
        total_false_negatives = sum([score["false_negatives"] for score in split_scores_by_document.values()])
        total_false_positives = sum([score["false_positives"] for score in split_scores_by_document.values()])

        # Calculate total measured score
        precision, recall, f1_score = Segmenter.calculate_measurement_scores(
            total_true_positives, total_false_negatives, total_false_positives)

        # Identify the document(s) with the worst performance in terms of score
        worst_scores = Segmenter.get_worst_scores(split_scores_by_document)

        print("===========================================================================\n" +
              ("Segmentation Error Analysis for %s:\n" % self.name) +
              "===========================================================================\n" +
              "True Positives: %d\nFalse Negatives: %d\nFalse Positives: %d\n" % (
                  total_true_positives, total_false_negatives, total_false_positives) +
              "Precision: %f\nRecall: %f\nF1 Score: %f\n" % (precision, recall, f1_score) +
              "The %s worst performing document(s) in terms of score: %s\n" % (
                  len(worst_scores), [document_id for document_id in worst_scores]) +
              ("The segmentation process took %s.\n" % duration) +
              "===========================================================================")

    def apply_segmentation(self, annotated=True, debug=False):
        """
        Generates sentences using Spacy

        :param annotated: Whether or not the data is annotated (default: True)
        :param debug: Whether or not an error analysis is performed for the segmentation (default: False)
        :return: Sentences generated by Spacy
        """
        log("Applying %s.." % self.name)

        # Start the timer
        start = datetime.datetime.now().replace(microsecond=0)

        # Generate Spacy sentences
        generated_sentences_by_document = {
            document_id: self.generate_sentences(plain_text)
            for document_id, plain_text in tqdm(self.get_plain_texts_by_document(annotated).items())
        }

        # End the timer and compute duration
        duration = datetime.datetime.now().replace(microsecond=0) - start

        # Analyze segmentation
        if annotated and debug:
            log("Analyzing %s.." % self.name)
            self.analyze_segmentation(generated_sentences_by_document, duration)

        log("%s resulted in %d sentences from %d documents!" % (
            self.name,
            sum([len(sentences) for sentences in generated_sentences_by_document.values()]),
            len(generated_sentences_by_document)
        ))

        # Return the resulting segmentation
        return generated_sentences_by_document
