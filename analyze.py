from corpus import Corpus


def analyze(corpus_fpath):
    # Initialize Corpus
    corpus = Corpus(corpus_fpath)
    print(corpus)


if __name__ == "__main__":
    analyze("./data/ldsi_w21_curated_annotations_v2.json")

