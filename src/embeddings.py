"""
 File:   embeddings.py
 Author: Batuhan Erden
"""

import fasttext

from src.utils.logging_utils import log


class Embeddings:

    NUM_NEIGHBORS_CONSIDERED_CLOSE = 10

    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the existing embeddings model

        :return: The existing embeddings model
        """
        model = None

        try:
            log("The embeddings model is being loaded..")
            model = fasttext.load_model(self.model_filepath)
            log("The embeddings model successfully loaded!")
        except ValueError:
            log("No embeddings model found!")

        return model

    def save_model(self):
        """
        Saves the embeddings model
        """
        log("The embeddings model is being saved..")
        self.model.save_model(self.model_filepath)
        log("The embeddings model successfully saved!")

    def train(self, tokens_filepath, num_epochs=20):
        """
        Trains the embeddings model and saves it

        :param tokens_filepath: The path to the file containing the tokens
        :param num_epochs: The number of epochs
        """
        log("Training the embeddings model for %d epochs.." % num_epochs)
        self.model = fasttext.train_unsupervised(input=tokens_filepath, model="skipgram", dim=100, min_count=3,
                                                 epoch=num_epochs)
        log("The embeddings model successfully trained!")

        self.save_model()

    def get_nearest_neighbors(self, words):
        """
        Gets NUM_NEIGHBORS_CONSIDERED_CLOSE nearest neighbors of words given

        :param words: Words to get the nearest neighbor information for
        :return: nearest neighbors of each word given
        """
        return {word: self.model.get_nearest_neighbors(word, k=self.NUM_NEIGHBORS_CONSIDERED_CLOSE) for word in words}
