import numpy as np
import pytest

import cnnclustering.cnn as cnn


class TestGetPoints:
    """Tests for data retrieval"""

    def by_parts_no_data(self, empty_cobj):
        part_iterator = empty_cobj.data.points.by_parts()
        with pytest.raises(StopIteration):
            next(part_iterator)

    def test_by_parts_std(self, std_cobj):
        part_iterator = std_cobj.data.points.by_parts()
        with pytest.raises(StopIteration):
            for _ in iter(int, 1):
                next(part_iterator)


class TestGetShape:
    """Tests for staticmethod :meth:core.cnn.Data.get_shape"""

    def test_None(self):
        d, e = cnn.Points.get_shape(None)
        assert d is None
        assert e == None

    def test_no_sequence(self):
        with pytest.raises(TypeError):
            d, e = cnn.Points.get_shape(1)

    def test_1point_1d(self):
        d, e = cnn.Points.get_shape([1])
        np.testing.assert_array_equal(d, np.array([[1]]))
        assert e == [1]

    def test_1point_2d(self):
        d, e = cnn.Points.get_shape([1, 2])
        np.testing.assert_array_equal(d, np.array([[1, 2]]))
        assert e == [1]

    def test_2points_3d(self):
        d, e = cnn.Points.get_shape([[1, 2, 3], [3, 2, 1]])
        np.testing.assert_array_equal(d, np.array([[1, 2, 3], [3, 2, 1]]))
        assert e == [2]

    def test_2parts_2_3points_4d(self):
        d, e = cnn.Points.get_shape(
            [[[1, 2, 3, 4], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]]
            )
        x = np.array([
                [1, 2, 3, 4], [4, 3, 2, 1],
                [1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]
                ])
        for c, i in enumerate(d):
            np.testing.assert_array_equal(i, x[c])

        assert e == [2, 3]

    def test_2points_2_3d(self):
        with pytest.raises(AssertionError):
            d, e = cnn.Points.get_shape([[1, 2], [3, 2, 1]])

    def test_2parts_2_2points_2_2_3_3d(self):
        with pytest.raises(AssertionError):
            d, e = cnn.Points.get_shape([
                [[1, 2], [3, 4]],
                [[1, 2, 3], [2, 3, 4]]
                ])

    @pytest.mark.skip(reason="This edge case is not caught yet")
    def test_2parts_2_2points_2_3_3_3d(self):
        with pytest.raises(AssertionError):
            d, e = cnn.Points.get_shape([
                [[1, 2], [3, 2, 1]],
                [[1, 2, 3], [2, 3, 4]]
                ])