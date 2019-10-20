from tqdm import tqdm
import numpy as np

from ..utils import sigmoid

__all__ = ['LinearRegression', 'RidgeRegression']


class LinearRegression(object):
    def __init__(self, fit_intercept: bool = True):
        """
        An ordinary least squares regression model fit via the normal equation.

        .. math::

            \mathbf{y} = \mathbf{X} \mathbf{w}

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an additional intercept term in addition to the
            model coefficients. Default is True.
        """
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the regression coefficients via maximum likelihood.

        .. math::

            \mathbf{w^{*}} = (X^{T} X)^{-1} X^{T} y

        Parameters
        ----------
        X : shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
        """

        # decide invertible or not
        self._invertible(X)

        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        pseudo_inverse = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)
        self.beta = np.dot(pseudo_inverse, y)

    def predict(self, X: np.array) -> np.array:
        """
        Used the trained model to generate predictions on a new collection of
        data points.

        Parameters
        ----------
        X : shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred :shape `(Z, K)`
            The model predictions for the items in `X`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)

    def _invertible(self, X: np.array):
        """
        Decide the X is invertible or not.

        Parameters
        ----------
        X : shape `(N, M)`
            A dataset consisting of `N` new examples, each of dimension `M`.

        """
        N = X.shape[0]
        M = X.shape[1]

        if self.fit_intercept:
            M = M + 1
        if not N >= M:
            raise Exception('The batch dimension `%d` is smaller than the feature dimension `%d`,'
                            'please use RidgeRegression instead.' % (N, M))


class RidgeRegression(object):
    def __init__(self, alpha=1, fit_intercept=True):
        """
        A ridge regression model fit via the normal equation.

        Parameters
        ----------
        alpha : float
            L2 regularization coefficient. Higher values correspond to larger
            penalty on the L2 norm of the model coefficients. Default is 1.
        fit_intercept : bool
            Whether to fit an additional intercept term in addition to the
            model coefficients. Default is True.
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the regression coefficients via maximum likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.dot(np.linalg.inv(X.T @ X + A), X.T)
        # pseudo_inverse = np.dot(np.linalg.pinv(X.T @ X + A), X.T)
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of
        data points.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The model predictions for the items in `X`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)
