"""
 File:   luima_law_segmenter.py
 Author: Batuhan Erden
"""

from luima_sbd.sbd_utils import text2sentences

from src.segmentation.segmenter import Segmenter


class LuimaLawSegmenter(Segmenter):

    def __init__(self, corpus):
        super().__init__(name="Luima Segmentation", corpus=corpus)

    def generate_sentences(self, plain_text):
        """
        Generates sentences from a plain text using a law-specific sentence segmenter (Luima)

        Overrides the definition in the super class

        :param plain_text: The plain text
        :return: Sentences generated from a plain text by Luima
        """

        indices = text2sentences(plain_text, offsets=True)

        return [
            {
                "txt": plain_text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
            }
            for (start_char, end_char) in indices
        ]

