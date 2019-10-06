import unittest

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from . import LinearRegression


class MyTestCase(unittest.TestCase):
    def test_linear_regression(self):
        test_times = 1000
        for _ in range(test_times):
            M = np.random.randint(1, 100)
            N = np.random.randint(1, 100)
            P = np.random.randint(1, 100)
            W = np.random.rand(M, P)
            X = np.random.rand(N, M)
            y = np.dot(X, W) + np.random.randn(N, P)
            fit_intercept = False

            mine = LinearRegression(fit_intercept=fit_intercept)

            if not N >= (M + int(fit_intercept)):
                self.assertRaises(Exception, mine.fit, X, y)
            else:
                mine.fit(X, y)
                mine_pre = mine.predict(X)
                their = linear_model.LinearRegression(fit_intercept=fit_intercept)
                their.fit(X, y)
                their_pre = their.predict(X)
                np.testing.assert_almost_equal(their_pre, mine_pre, decimal=3)


if __name__ == '__main__':
    unittest.main()
