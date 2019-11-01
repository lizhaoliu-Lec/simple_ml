import pickle
import copy
import numpy as np

from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier:
    """A simple AdaBoost Classifier."""

    def __init__(self, weak_classifier: DecisionTreeClassifier, n_weakers_limit):
        """Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        """
        self.n_weakers_limit = n_weakers_limit
        self.classifiers = [copy.deepcopy(weak_classifier) for _ in range(n_weakers_limit)]

    def is_good_enough(self):
        """Optional"""
        pass

    def fit(self, X, y, eps=1e-8):
        """Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        """
        N = X.shape[0]
        ws = np.ones(N) / N
        for idx in range(self.n_weakers_limit):
            # (1) fit weak classifier
            self.classifiers[idx].fit(X, y)
            # (2) compute error rate
            y_pred = self.classifiers[idx].predict(X)
            error_rate = np.sum(ws * (y != y_pred).float()) + eps
            # (3) compute the coefficient of the idx classifier
            alpha_idx = 0.5 * np.log((1 - error_rate) / error_rate)
            # (4) update weights
            z_idx = None
            #TODO

    def predict_scores(self, X):
        """Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        """
        pass

    def predict(self, X, threshold=0):
        """Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        """
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
