"""
File:   corpus.py
Author: Batuhan Erden
"""

import json
import glob
from tqdm import tqdm

from src.utils import data_utils
from src.utils.sys_utils import log


class Corpus:

    CASE_HEADER_TYPE = "Header"

    def __init__(self, annotations_filepath, unlabeled_data_dir, sparse_corpus=False):
        if sparse_corpus:  # Do not load anything for sparse corpus
            self.train_spans, self.val_spans, self.test_spans = [], [], []
            return
        
        # Load labeled data
        self.annotated_documents_by_id, self.annotations_by_document, self.types_by_id = \
            Corpus.load_labeled_data(annotations_filepath)

        # Read unlabeled data
        self.unlabeled_data = Corpus.load_unlabeled_data(unlabeled_data_dir)

        # Create balanced split spans
        self.train_spans, self.val_spans, self.test_spans = self.create_balanced_span_splits()

    @staticmethod
    def create_annotated_documents_by_id(documents_by_id, annotations):
        """
        Creates annotated documents by ID

        :param documents_by_id: All documents by ID
        :param annotations: All annotations
        :return: Annotated documents by ID
        """
        annotated_documents_by_id = {}

        for annotation in annotations:
            document_id = annotation["document"]
            annotated_documents_by_id[document_id] = documents_by_id[document_id]

        return annotated_documents_by_id

    @staticmethod
    def create_annotations_by_document(annotations):
        """
        Maps documents to their annotations for easy access (e.g.: { "doc_1_id": [ ... ], "doc_2_id": [ ... ], ... })

        :param annotations: All annotations
        :return: Annotations by document
        """
        annotations_by_document = {}

        for annotation in annotations:
            document_id = annotation["document"]

            if document_id not in annotations_by_document:
                annotations_by_document[document_id] = []

            annotations_by_document[document_id].append(annotation)

        return annotations_by_document

    @staticmethod
    def load_labeled_data(annotations_filepath):
        """
        Loads the labeled data with annotations

        :param annotations_filepath: Path to the file containing the annotations
        :return: A tuple containing the data (annotated_documents_by_id, annotations_by_document, types_by_id)
        """
        log("Loading the labeled data..")

        # Load the labeled data
        data = json.load(open(annotations_filepath))

        # Distribute the data into dictionaries
        documents_by_id = {d["_id"]: d for d in data["documents"]}
        types_by_id = {t["_id"]: t for t in data["types"]}

        # Only work with annotated documents
        annotated_documents_by_id = Corpus.create_annotated_documents_by_id(documents_by_id, data["annotations"])
        annotations_by_document = Corpus.create_annotations_by_document(data["annotations"])

        log("%d documents loaded, %d of which are annotated!" %
            (len(documents_by_id), len(annotated_documents_by_id)))
        return annotated_documents_by_id, annotations_by_document, types_by_id

    @staticmethod
    def load_unlabeled_data(unlabeled_data_dir):
        """
        Reads unlabeled data

        :param unlabeled_data_dir: Directory containing the unlabeled data
        :return: Unlabeled data
        """
        log("Loading the unlabeled data..")
        unlabeled = {}

        # Load the unlabeled data
        for file in tqdm(glob.glob(unlabeled_data_dir + "*.txt")):
            with open(file, encoding="latin-1") as data:
                unlabeled[file] = data.read()

        log("%d unlabeled documents loaded!" % len(unlabeled))
        return unlabeled

    @staticmethod
    def create_span(plainText, txt, start, end, document_id=None, span_type=None, outcome=None):
        return {
            "txt": txt,
            "document": document_id,
            "type": span_type,
            "start": start,
            "start_normalized": start / len(plainText),
            "end": end,
            "outcome": outcome
        }

    def create_span_data(self, annotated_document_ids):
        """
        Gets all sentences assuming every annotation is a sentence

        :param annotated_document_ids: Annotated document IDs used to create span data
        :return: Span data created from the documents given
        """
        span_data = []

        # Create span data for each annotated document
        for document_id in annotated_document_ids:
            document_span_data = []

            document = self.annotated_documents_by_id[document_id]
            annotations = self.annotations_by_document[document_id]

            for annotation in annotations:
                annotation_start = annotation["start"]
                annotation_end = annotation["end"]

                document_span_data.append(Corpus.create_span(
                    plainText=document["plainText"], txt=document["plainText"][annotation_start:annotation_end],
                    start=annotation_start, end=annotation_end, document_id=document_id,
                    span_type=self.types_by_id[annotation["type"]]["name"], outcome=document["outcome"]
                ))

            # Sort document span data by annotation start
            span_data += sorted(document_span_data, key=lambda span: span["start"])

        return span_data

    def create_balanced_span_splits(self):
        """
        Creates spans (train, val and test) from balanced split documents

        :return: Train, validation and test spans
        """
        # Split data in a balanced way
        document_ids_train, document_ids_val, document_ids_test = data_utils.split_data_balanced(
            inputs=self.annotated_documents_by_id, type_key="outcome", val_test_size=.1)

        # Create train, val and test spans from documents split in a balanced way
        train_spans = self.create_span_data(document_ids_train)
        val_spans = self.create_span_data(document_ids_val)
        test_spans = self.create_span_data(document_ids_test)

        # Check if the data has been split in a balanced way
        data_utils.assert_balanced_split((train_spans, val_spans, test_spans), "document", "outcome")

        return train_spans, val_spans, test_spans

    @staticmethod
    def get_documents_split(spans):
        """
        Gets distinct documents given spans

        :param spans: Spans given
        :return: Distinct documents given spans
        """
        return list(set([span["document"] for span in spans]))

    def get_plain_texts_by_annotated_document(self, spans):
        """
        Gets the plain text of each annotated document

        :param spans: Spans (train, test or validation)
        :return: The plain text of each annotated document
        """
        return {
            document_id: self.annotated_documents_by_id[document_id]["plainText"]
            for document_id in Corpus.get_documents_split(spans)
        }

    def get_distinct_headers(self, spans):
        """
        Returns distinct headers (e.g. "INTRODUCTION", "REPRESENTATION", etc.)

        :return: Distinct headers
        """
        return list(set([
            span["txt"] for span in spans
            if span["type"] == self.CASE_HEADER_TYPE
        ]))

    def __str__(self):
        val_documents = self.get_documents_split(self.val_spans)
        test_documents = self.get_documents_split(self.test_spans)

        return "===========================================================================\n" + \
               "Corpus summary:\n" + \
               "===========================================================================\n" + \
               "Validation set: %s\nTest set: %s" % (val_documents, test_documents) + "\n" + \
               "==========================================================================="
