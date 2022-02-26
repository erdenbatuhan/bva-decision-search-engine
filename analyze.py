import spacy
import json
import random
import re
import pandas as pd
import numpy as np
from copy import deepcopy

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.symbols import ORTH

from sklearn import model_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

import matplotlib.pyplot as plt

from corpus import Corpus


def analyze(corpus_fpath):
    # Initialize Corpus
    corpus = Corpus(corpus_fpath)
    print(corpus)


if __name__ == "__main__":
    analyze("./data/ldsi_w21_curated_annotations_v2.json")

