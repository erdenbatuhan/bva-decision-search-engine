from src.featurization.tfidf_featurizater import TFIDFFeaturizater


class WordEmbeddingFeaturizater(TFIDFFeaturizater):

    def __init__(self, corpus, tokenization_segmenter):
        super().__init__(corpus, tokenization_segmenter)

