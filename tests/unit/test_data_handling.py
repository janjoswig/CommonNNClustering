import numpy as np
import pytest

import core.cnn as cnn


class TestGetPoints:
    """Tests for data retrieval"""

    def by_parts_no_data(self, empty_cobj):
        part_iterator = empty_cobj.data.by_parts()
        with pytest.raises(StopIteration):
            next(part_iterator)

    def test_by_parts_std(self, std_cobj):
        part_iterator = std_cobj.data.by_parts()
        with pytest.raises(StopIteration):
            for _ in iter(int, 1):
                next(part_iterator)


class TestGetShape:
    """Tests for staticmethod :meth:core.cnn.Data.get_shape"""

    def test_None(self):
        d, s = cnn.Data.get_shape(None)
        assert d is None
        assert s == {"parts": None, "points": None, "dimensions": None}

    def test_no_sequence(self):
        with pytest.raises(TypeError):
            d, s = cnn.Data.get_shape(1)

    def test_1point_1d(self):
        d, s = cnn.Data.get_shape([1])
        np.testing.assert_array_equal(d, np.array([[1]]))
        assert s == {"parts": 1, "points": (1, ), "dimensions": 1, }

    def test_1point_2d(self):
        d, s = cnn.Data.get_shape([1, 2])
        np.testing.assert_array_equal(d, np.array([[1, 2]]))
        assert s == {"parts": 1, "points": (1, ), "dimensions": 2, }

    def test_2points_3d(self):
        d, s = cnn.Data.get_shape([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(d, np.array([[1, 2, 3], [3, 2, 1]]))
        assert s == {"parts": 1, "points": (2, ), "dimensions": 3, }

    def test_2_parts_2_3points_4d(self):
        d, s = cnn.Data.get_shape(
            [[[1, 2, 3, 4], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]]
            )
        e = np.array([
                [1, 2, 3, 4], [4, 3, 2, 1],
                [1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]
                ])
        for c, i in enumerate(d):
            np.testing.assert_array_equal(i, e[c])

        assert s == {"parts": 2, "points": (2, 3), "dimensions": 4, }

    def test_2points_2_3d(self):
        with pytest.raises(IndexError):
            d, s = cnn.Data.get_shape([[1, 2], [3, 2, 1]])