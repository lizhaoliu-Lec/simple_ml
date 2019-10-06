import numpy as np

__all__ = ['euclidean']


def euclidean(x: np.array, y: np.array) -> float:
    """
    Compute the Euclidean (`L2`) distance between two real vectors

    Notes
    -----
    The Euclidean distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_i (x_i - y_i)^2  }

    Parameters
    ----------
    x,y : shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L2 distance between **x** and **y**.
    """
    return np.sqrt(np.sum((x - y) ** 2))
