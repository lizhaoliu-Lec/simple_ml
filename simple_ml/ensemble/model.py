import pickle
import copy
from tqdm import tqdm
import numpy as np


class AdaBoostClassifier:
    """A simple AdaBoost Classifier."""

    def __init__(self, weak_classifier, n_weakers_limit, patience=5):
        """Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
            patience: How many 'patience' we should wait before early stopping.
        """
        self.n_weakers_limit = n_weakers_limit
        self.classifiers = [copy.deepcopy(weak_classifier) for _ in range(n_weakers_limit)]
        self.best_n_weakers_limit = n_weakers_limit
        self.alphas = None
        self.performance_history = []
        self.patience = patience

    def _is_good_enough(self, X_val, y_val, m_step, threshold):
        """Optional"""
        scores = None
        for m in range(m_step + 1):
            ym = self.alphas[m] * self.classifiers[m].predict(X_val)
            ym = np.reshape(ym, (-1, 1))
            if scores is None:
                scores = ym
            else:
                scores += ym
        positive_mask = (scores >= threshold)
        negative_mask = (scores < threshold)
        y_val_pred = positive_mask * 1 + negative_mask * -1
        acc = np.mean(y_val == y_val_pred)
        if len(self.performance_history) == 0:
            self.performance_history.append(acc)
            return False
        else:
            if self.performance_history[-1] > acc:
                if self.patience == 0:
                    return True
                else:
                    self.performance_history.append(acc)
                    self.patience = self.patience - 1
                    return False
            else:
                self.performance_history.append(acc)
                return False

    def fit(self, X, y, X_val=None, y_val=None, early_stop=False, eps=1e-8, threshold=0):
        """Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
            X_val: Same type and constraint as X, but use for validation.
            y_val: Same type and constraint as y, but use for validation.
            early_stop: Whether to early stop if we find a promising number of weak classifier.
            eps: Very small float for numeric stable.
            threshold: Only used to perform validation.
        """
        N = X.shape[0]
        wm = np.ones((N, 1)) / N
        self.alphas = np.zeros((self.n_weakers_limit, 1))
        for m in tqdm(range(self.n_weakers_limit)):
            # (1) fit weak classifier
            self.classifiers[m].fit(X, y, sample_weight=np.squeeze(wm, axis=1))
            # (2) compute error rate
            y_pred = self.classifiers[m].predict(X)
            y_pred = np.reshape(y_pred, (-1, 1))
            error_rate = np.sum(wm * (y != y_pred)).clip(min=eps, max=1-eps)
            # if the classifier is not good enough, we ignore it
            if error_rate > 0.5:
                continue
            # (3) compute the coefficient of the m classifier
            alpha_m = 0.5 * np.log((1 - error_rate) / error_rate)
            self.alphas[m] = alpha_m
            # (4) update weights
            wm = wm * np.exp(-1 * alpha_m * y * y_pred)
            wm = wm / np.sum(wm)
            # (5) see if good enough, good enough will stop training one more classifier
            if early_stop and self._is_good_enough(X_val, y_val, m, threshold):
                print('The best number of weak learner is `%d`, early stop perform!')
                self.best_n_weakers_limit = m
                break

    def predict_scores(self, X):
        """Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of different samples, which shape should be (n_samples,1).
        """
        scores = None
        if np.sum(self.alphas) == 0.0:
            raise ValueError('All the weak learner are ignored during training, perhaps change a weak learner.')
        for m in range(self.best_n_weakers_limit):
            ym = self.alphas[m] * self.classifiers[m].predict(X)
            ym = np.reshape(ym, (-1, 1))
            if scores is None:
                scores = ym
            else:
                scores += ym
        return scores

    def predict(self, X, threshold=0):
        """Predict the categories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of dividing the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        """
        y_pred = self.predict_scores(X)
        positive_mask = (y_pred >= threshold)
        negative_mask = (y_pred < threshold)
        return positive_mask * 1 + negative_mask * -1

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
