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
            log("The embeddings model is successfully loaded!")
        except ValueError:
            log("No embeddings model found, please train a new one!")

        return model

    def save_model(self):
        """
        Saves the embeddings model
        """

        log("The embeddings model is being saved..")
        self.model.save_model(self.model_filepath)
        log("The embeddings model is successfully saved!")

    def train(self, tokens_filepath, model_type="skipgram", model_dim=100, model_min_count=20, model_epoch=20):
        """
        Trains the embeddings model with model arguments
        """

        log("Training the embeddings model for %d epochs.." % model_epoch)
        self.model = fasttext.train_unsupervised(input=tokens_filepath, model=model_type,
                                                 dim=model_dim, min_count=model_min_count,
                                                 epoch=model_epoch)
        log("The embeddings model is successfully trained!")

        self.save_model()

    def get_nearest_neighbors(self, words):
        """
        Gets NUM_NEIGHBORS_CONSIDERED_CLOSE nearest neighbors of words given

        :param words: Words to get the nearest neighbor information for
        :return: nearest neighbors of each word given
        """

        return {word: self.model.get_nearest_neighbors(word, k=self.NUM_NEIGHBORS_CONSIDERED_CLOSE) for word in words}

