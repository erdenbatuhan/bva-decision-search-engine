import fasttext

from src.utils.logging_utils import log


class Embeddings:

    def __init__(self, model_filepath, tokens_filepath, train_new=False,
                 model_type="skipgram", model_dim=100, model_minn=100, model_epoch=50):
        self.model_filepath = model_filepath
        self.model = self.load_model() if not train_new else None
        self.model_args = {
            "input": tokens_filepath,
            "model": model_type,
            "dim": model_dim,
            "minn": model_minn,
            "epoch": model_epoch
        }

    def load_model(self):
        """
        Loads the existing embeddings model

        :return: The existing embeddings model
        """

        log("The embeddings model is being loaded..")
        model = fasttext.load_model(self.model_filepath)
        log("The embeddings model is successfully loaded!")

        return model

    def save_model(self):
        """
        Saves the embeddings model
        """

        log("The embeddings model is being saved..")
        self.model.save_model(self.model_filepath)
        log("The embeddings model is successfully saved!")

    def train(self):
        """
        Trains the embeddings model with model arguments
        """

        log("Training the embeddings model for %d epochs.." % self.model_args["epoch"])

        self.model = fasttext.train_unsupervised(input=self.model_args["input"], model=self.model_args["model"],
                                                 dim=self.model_args["dim"], minn=self.model_args["minn"],
                                                 epoch=self.model_args["minn"])
        self.save_model()

        log("The embeddings model is successfully trained!")

