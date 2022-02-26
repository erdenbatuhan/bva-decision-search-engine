import spacy

from src.segmentation.segmenter import Segmenter


class SpacySegmenter(Segmenter):

    def __init__(self, corpus, improved=False):
        super().__init__(corpus=corpus)

        self.improved = improved

        # Load basic English pipeline provided by spacy
        self.nlp = spacy.load("en_core_web_sm")

        if improved:  # Introduce additional exceptions/extensions
            self.extend_language()

    def extend_language(self):
        """
        Introduces additional exceptions/extensions to the language/segmenter
        """

        # TODO: Add some constraints here
        pass

    def apply_segmentation(self):
        """
        Generates sentences using Spacy

        :return: Sentences generated by Spacy
        """

        # Generate spacy sentences
        generated_sentences_by_document = {
            document_id: list(self.nlp(self.corpus.annotated_documents_by_id[document_id]["plainText"]).sents)
            for document_id in self.corpus.get_documents_split(self.corpus.train_spans)
        }

        # Analyze segmentation
        self.analyze_segmentation("Spacy Segmentation (%s)" % ("Naive" if not self.improved else "Improved"),
                                  generated_sentences_by_document)

        return generated_sentences_by_document

