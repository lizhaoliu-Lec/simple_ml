import unittest
import numpy as np

from sklearn.preprocessing import StandardScaler

from .general import Standardizer


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_standardizer():
        test_times = 10000
        for _ in range(test_times):
            mean = bool(np.random.randint(2))
            std = bool(np.random.randint(2))
            N = np.random.randint(2, 100)
            M = np.random.randint(2, 100)
            X = np.random.rand(N, M)

            S = Standardizer(with_mean=mean, with_std=std)
            S.fit(X)
            mine = S.transform(X)

            theirs = StandardScaler(with_mean=mean, with_std=std)
            gold = theirs.fit_transform(X)

            np.testing.assert_almost_equal(mine, gold)


if __name__ == '__main__':
    unittest.main()
