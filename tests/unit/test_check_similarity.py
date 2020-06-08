"""Testing functions checking the CNN density criterion

Do two collections of point indices have a minimum number of common
elements?
"""

import numpy as np

import cnnclustering.cnn as cnn
import cnnclustering._cfits as cfits
import cnnclustering._fits as fits

CASES = [  # collection a, collection b, common elements c, result
        ([0, 1, 2, 4], [1, 2, 5, 6], 2, True),
        ([0, 1, 2, 4], [1, 2, 5, 6], 3, False),
        ([], [], 0, True),
        ([], [], 1, False),
        ([0, 1, 2], [2, 3], 0, True),
        ([0, 1, 2], [3, 4], 0, True)
        ]


class TestPython:
    def test_check_similarity_array(self):
        for a, b, c, result in CASES:
            a = np.asarray(a)
            b = np.asarray(b)
            if result is True:
                assert fits.check_similarity_array(a, b, c)
            else:
                assert not fits.check_similarity_array(a, b, c)

    def test_check_similarity_set(self):
        for a, b, c, result in CASES:
            a = set(a)
            b = set(b)
            if result is True:
                assert fits.check_similarity_set(a, b, c)
            else:
                assert not fits.check_similarity_set(a, b, c)

    def test_check_similarity_list(self):
        for a, b, c, result in CASES:
            if result is True:
                assert fits.check_similarity_collection(a, b, c)
            else:
                assert not fits.check_similarity_collection(a, b, c)


class TestCython:
    def test_check_similarity_set(self):
        for a, b, c, result in CASES:
            a = set(a)
            b = set(b)
            if result is True:
                assert cfits._check_similarity_set(a, b, c)
            else:
                assert not cfits._check_similarity_set(a, b, c)

    def test_check_similarity_cppset(self):
        for a, b, c, result in CASES:
            a = set(a)
            b = set(b)
            if result is True:
                assert cfits._check_similarity_cppset(a, b, c)
            else:
                assert not cfits._check_similarity_cppset(a, b, c)

    def test_check_similarity_array(self):
        for a, b, c, result in CASES:
            a = np.asarray(a, dtype=np.intp)
            b = np.asarray(b, dtype=np.intp)
            if result is True:
                assert cfits._check_similarity_array(a, b, c)
            else:
                assert not cfits._check_similarity_array(a, b, c)
