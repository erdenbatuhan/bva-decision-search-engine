"""
 File:   corpus.py
 Author: Batuhan Erden
"""

import json

from src.utils import data_utils


class Corpus:

    def __init__(self, corpus_fpath):
        # Load corpus data
        self.annotated_documents_by_id, self.annotations_by_document, self.types_by_id = \
            self.load_corpus_data(corpus_fpath)

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

    def load_corpus_data(self, corpus_fpath):
        """
        Loads corpus data

        :param corpus_fpath: The path of the corpus data
        :return: A tuple containing the data (annotated_documents_by_id, annotations_by_document, types_by_id)
        """

        # Load data
        data = json.load(open(corpus_fpath))

        # Distribute the data into dictionaries
        documents_by_id = {d["_id"]: d for d in data["documents"]}
        types_by_id = {t["_id"]: t for t in data["types"]}

        # Only work with annotated documents
        annotated_documents_by_id = self.create_annotated_documents_by_id(documents_by_id, data["annotations"])
        annotations_by_document = self.create_annotations_by_document(data["annotations"])

        return annotated_documents_by_id, annotations_by_document, types_by_id

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

                document_span_data.append({
                    "txt": document["plainText"][annotation_start:annotation_end],
                    "document": document_id,
                    "type": self.types_by_id[annotation["type"]]["name"],
                    "start": annotation_start,
                    "start_normalized": annotation_start / len(document["plainText"]),
                    "end": annotation_end,
                    "outcome": document["outcome"]
                })

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

    @staticmethod
    def get_documents_split(spans):
        """
        Gets distinct documents given spans

        :param spans: Spans given
        :return: Distinct documents given spans
        """

        return list(set([span["document"] for span in spans]))

    def __str__(self):
        val_documents = self.get_documents_split(self.val_spans)
        test_documents = self.get_documents_split(self.test_spans)

        return "===========================================================================\n" + \
               "Corpus summary:\n" + \
               "===========================================================================\n" + \
               "Validation set: %s\nTest set: %s" % (val_documents, test_documents) + "\n" + \
               "==========================================================================="

