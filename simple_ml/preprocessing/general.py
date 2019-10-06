import numpy as np

__all__ = ['Standardizer']


class Standardizer(object):
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Feature-wise standardization for vector inputs.

        Notes
        -----
        Due to the sensitivity of empirical mean and standard deviation
        calculations to extreme values, `Standardizer` cannot guarantee
        balanced feature scales in the presence of outliers. In particular,
        note that because outliers for each feature can have different
        magnitudes, the spread of the transformed data on each feature can be
        very different.

        Similar to sklearn, `Standardizer` uses a biased estimator for the
        standard deviation: ``numpy.std(x, ddof=0)``.

        Parameters
        ----------
        with_mean :
            Whether to scale samples to have 0 mean during transformation.
            Default is True.
        with_std :
            Whether to scale samples to have unit variance during
            transformation. Default is True.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self._is_fit = False

        self._mean = None
        self._std = None

    @property
    def hyper_parameters(self) -> dict:
        h = {
            "with_mean": self.with_mean,
            "with_std": self.with_std
        }
        return h

    @property
    def parameters(self) -> dict:
        params = {
            "mean": self._mean,
            "std": self._std,
        }
        return params

    def __call__(self, X) -> np.array:
        return self.transform(X)

    def fit(self, X: np.array):
        """
        Store the feature-wise mean and standard deviation across the samples
        in `X` for future scaling.

        Parameters
        ----------
        X: An array of N samples, each with dimensionality `C`
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[0] < 2:
            raise ValueError("`X` must contain at least 2 samples")

        std = np.ones(X.shape[1])
        mean = np.zeros(X.shape[1])

        if self.with_mean:
            mean = np.mean(X, axis=0)

        if self.with_std:
            std = np.std(X, axis=0, ddof=0)

        self._mean = mean
        self._std = std
        self._is_fit = True

    def transform(self, X: np.array) -> np.array:
        """
        Standardize features by removing the mean and scaling to unit variance.

        For a sample `x`, the standardized score is calculated as:

        .. math::

            z = (x - u) / s

        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.

        Parameters
        ----------
        X : shape `(N, C)`
            An array of N samples, each with dimensionality `C`.

        Returns
        -------
        Z : shape `(N, C)`
            The feature-wise standardized version of `X`.
        """
        if not self._is_fit:
            raise Exception("Must call `fit` before using the `transform` method")
        return (X - self._mean) / self._std

    def inverse_transform(self, Z: np.array) -> np.array:
        """
        Convert a collection of standardized features back into the original
        feature space.

        For a standardized sample `z`, the unstandardized score is calculated as:

        .. math::

            x = z s + u

        where `u` is the mean of the training samples or zero if `with_mean` is
        False, and `s` is the standard deviation of the training samples or 1
        if `with_std` is False.

        Parameters
        ----------
        Z : shape `(N, C)`
            An array of `N` standardized samples, each with dimensionality `C`.

        Returns
        -------
        X : shape `(N, C)`
            The unstandardixed samples from `Z`.
        """
        assert self._is_fit, "Must fit `Standardizer` before calling inverse_transform"
        P = self.parameters
        mean, std = P["mean"], P["std"]
        return Z * std + mean
