from sklearn.feature_extraction.text import TfidfVectorizer

from src.tokenizer import Tokenizer
from src.utils.logging_utils import log


class TFIDFFeaturizater:

    MIN_TOKEN_FREQUENCY = 3
    NGRAM_RANGE = (1, 1)

    def __init__(self, corpus, tokenization_segmenter):
        self.corpus = corpus
        self.span_texts = {
            "train": [span['txt'] for span in corpus.train_spans],
            "test": [span['txt'] for span in corpus.test_spans],
            "val": [span['txt'] for span in corpus.val_spans]
        }

        self.tokenization_segmenter = tokenization_segmenter
        self.tfidf_vectorizer = None

    def tokenize(self, plain_text):
        return Tokenizer.tokenize(self.tokenization_segmenter, plain_text)

    def fit(self):
        log("Vectorizing the training data using TfidfVectorizer..")

        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize,
                                                min_df=self.MIN_TOKEN_FREQUENCY,
                                                ngram_range=self.NGRAM_RANGE)
        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(self.span_texts["train"])

        log("The training data vectorized successfully using TfidfVectorizer!")

    def get_vectorizer_feature_names(self):
        return self.tfidf_vectorizer.get_feature_names()

