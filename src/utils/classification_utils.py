"""
File:   classification_utils.py
Author: Batuhan Erden
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(true_labels, predicted_labels, classes, title=None, cmap=plt.cm.Blues):
    """
    Plots a beautiful confusion matrix

    Referenced from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    :param true_labels: Ground truth
    :param predicted_labels: Prediction
    :param classes: Classes
    :param title: Title of the plot (default: None)
    :param cmap: Color map of the plot (default: plt.cm.Blues)
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    return ax


def test_classifier(classifier, X, y, use_test_set=False):
    """
    Tests the classifier

    :param classifier: The classifier
    :param X: The inputs
    :param y: The labels
    :param use_test_set: When set to True, the test set is used for validation.
                         Otherwise, the val set is used (default: False)
    """
    # Train classifiers on train set and validate on val set
    for featurizer_name in X:
        # Train on train set
        classifier_trained = classifier.fit(X[featurizer_name]["train"], y[featurizer_name]["train"])

        # Validate on both test and val sets
        for dataset_type in ["train", "test" if use_test_set else "val"]:
            true_labels = y[featurizer_name][dataset_type]
            predicted_labels = classifier_trained.predict(X[featurizer_name][dataset_type])

            print(f"[{featurizer_name}] {dataset_type}:\n" +
                  f"{classification_report(true_labels, predicted_labels)}")

            plot_confusion_matrix(true_labels, predicted_labels,
                                  classes=list(classifier.classes_),
                                  title=f"{[featurizer_name]} {type(classifier_trained).__name__} ({dataset_type} set)")
            plt.show()

        print("-" * 100)  # Just a line separator
