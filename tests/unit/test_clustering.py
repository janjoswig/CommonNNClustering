import numpy as np

import cnnclustering.cnn as cnn
import cnnclustering._cfits as cfits


class TestReel:
    """Tests for child cluster merging after hierarchical clustering"""

    def test_reel_deep_None(self, hierarchical_cobj):
        hierarchical_cobj.reel(deep=None)
        np.testing.assert_array_equal(
            hierarchical_cobj.labels,
            cnn.Labels([0, 0, 0, 3, 0, 0, 0, 2, 6, 5, 0, 2, 2, 3, 0])
            )

    def test_reel_deep_1(self, hierarchical_cobj):
        hierarchical_cobj.reel(deep=1)
        np.testing.assert_array_equal(
            hierarchical_cobj.labels,
            cnn.Labels([0, 0, 0, 3, 0, 0, 0, 2, 4, 4, 4, 2, 2, 3, 0])
            )


class TestBFS:
    """Test BFS connected component search"""

    def test_array(self):
        edges = np.array([1, 2, 3,  # 0
                          0, 2, 4,  # 1
                          0, 1, 3,  # 2
                          0, 2,     # 3
                          1,        # 4
                          6,        # 5
                          5,        # 6
                          9, 10,    # 8
                          8, 10,    # 9
                          8, 9], dtype=np.uintp)
        indices = np.array(
            [0, 3, 6, 9, 11, 12, 13, 14, 14, 16, 18, 20], dtype=np.uintp
            )

        labels = np.asarray(cfits.fit_from_SparsegraphArray(edges, indices))
        np.testing.assert_array_equal(
            labels,
            np.array([1, 1, 1, 1, 1, 2, 2, 0, 3, 3, 3])
            )


class TestCython:

    def test_fit_from_PointsArray_base_points_e15_0(
            self,
            base_points,
            base_labels_e15_0):
        points = np.array(base_points, dtype=np.float_)
        labels = np.zeros(points.shape[0], dtype=np.int_)
        cfits.fit_from_PointsArray(points, labels, 1.5, 0)

        np.testing.assert_array_equal(
            np.array(base_labels_e15_0),
            labels
            )

    def test_fit_from_PointsArray_base_points_e15_1(
            self,
            base_points,
            base_labels_e15_1):
        points = np.array(base_points, dtype=np.float_)
        labels = np.zeros(points.shape[0], dtype=np.int_)
        cfits.fit_from_PointsArray(points, labels, 1.5, 1)

        np.testing.assert_array_equal(
            np.array(base_labels_e15_1),
            labels
            )

    def test_fit_from_PointsArray_base_points_e15_2(
            self,
            base_points,
            base_labels_e15_2):
        points = np.array(base_points, dtype=np.float_)
        labels = np.zeros(points.shape[0], dtype=np.int_)
        cfits.fit_from_PointsArray(points, labels, 1.5, 2)

        np.testing.assert_array_equal(
            np.array(base_labels_e15_2),
            labels
            )