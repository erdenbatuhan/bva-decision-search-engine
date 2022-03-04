import fasttext


class Embeddings:

    def __init__(self, model_filepath, tokens_filepath, train_new=False,
                 model_type="skipgram", model_dim=100, model_minn=100, model_epoch=30):
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

        return fasttext.load_model(self.model_filepath)

    def save_model(self):
        """
        Saves the embeddings model
        """

        self.model.save_model(self.model_filepath)

    def train(self):
        """
        Trains the embeddings model with model arguments
        """

        self.model = fasttext.train_unsupervised(input=self.model_args["input"], model=self.model_args["model"],
                                                 dim=self.model_args["dim"], minn=self.model_args["minn"],
                                                 epoch=self.model_args["minn"])

