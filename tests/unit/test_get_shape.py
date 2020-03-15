import numpy as np
import pytest


class TestGetShape:
    def test_None(self, empty_cobj):
        d, s = empty_cobj.get_shape(None)
        assert d is None
        assert s == {"parts": None, "points": None, "dimensions": None, }

    def test_no_sequence(self, empty_cobj):
        with pytest.raises(TypeError):
            d, s = empty_cobj.get_shape(1)

    def test_1point_1d(self, empty_cobj):
        d, s = empty_cobj.get_shape([1])
        np.testing.assert_array_equal(d, np.array([[[1]]]))
        assert s == {"parts": 1, "points": [1], "dimensions": 1, }

    def test_1point_2d(self, empty_cobj):
        d, s = empty_cobj.get_shape([1, 2])
        np.testing.assert_array_equal(d, np.array([[[1, 2]]]))
        assert s == {"parts": 1, "points": [1], "dimensions": 2, }

    def test_2points_3d(self, empty_cobj):
        d, s = empty_cobj.get_shape([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(d, np.array([[[1, 2, 3], [3, 2, 1]]]))
        assert s == {"parts": 1, "points": [2], "dimensions": 3, }

    def test_2_parts_2_3points_4d(self, empty_cobj):
        d, s = empty_cobj.get_shape(
            [[[1, 2, 3, 4], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]]
            )
        e = np.array([
                np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
                np.array([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]])
                ])
        for c, i in enumerate(d):
            np.testing.assert_array_equal(i, e[c])

        assert s == {"parts": 2, "points": [2, 3], "dimensions": 4, }

    def test_2points_2_3d(self, empty_cobj):
        with pytest.raises(IndexError):
            d, s = empty_cobj.get_shape([[1, 2], [3, 2, 1]])