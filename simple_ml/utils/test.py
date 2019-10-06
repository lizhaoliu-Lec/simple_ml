import unittest

import numpy as np
from scipy.spatial import distance

from . import euclidean


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_euclidean():
        test_times = 1000
        for _ in range(test_times):
            N = np.random.randint(1, 100)
            x = np.random.rand(N)
            y = np.random.rand(N)
            mine = euclidean(x, y)
            theirs = distance.euclidean(x, y)
            np.testing.assert_almost_equal(mine, theirs)


if __name__ == '__main__':
    unittest.main()
