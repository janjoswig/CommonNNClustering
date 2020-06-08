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

    def test_fit_from_PointsArray_base_points_e15(
            self,
            base_points,
            base_labels_e15_0,
            base_labels_e15_1,
            base_labels_e15_2):
        points = np.array(base_points, dtype=np.float_)
        cases = [(0, base_labels_e15_0),
                 (1, base_labels_e15_1),
                 (2, base_labels_e15_2)]
        for c, reflabels in cases:
            labels = np.zeros(points.shape[0], dtype=np.int_)
            consider = np.ones_like(labels, dtype=np.uint8)
            cfits.fit_from_PointsArray(points, labels, consider, 1.5, c)

            np.testing.assert_array_equal(
                np.array(reflabels),
                labels
                )

    def test_fit_from_DistancesArray_base_points_e15(
            self,
            base_distances,
            base_labels_e15_0,
            base_labels_e15_1,
            base_labels_e15_2):
        distances = np.array(base_distances, dtype=np.float_)
        cases = [(0, base_labels_e15_0),
                 (1, base_labels_e15_1),
                 (2, base_labels_e15_2)]
        for c, reflabels in cases:
            labels = np.zeros(distances.shape[0], dtype=np.int_)
            consider = np.ones_like(labels, dtype=np.uint8)
            cfits.fit_from_DistancesArray(distances, labels, consider, 1.5, c)

            np.testing.assert_array_equal(
                np.array(reflabels),
                labels
                )

    def test_fit_from_NeighbourhoodsList_base_neighbourhoods_e15(
            self,
            base_neighbourhoods_e15,
            base_labels_e15_0,
            base_labels_e15_1,
            base_labels_e15_2):
        neighbourhoods = [set(x) for x in base_neighbourhoods_e15]
        cases = [(0, base_labels_e15_0),
                 (1, base_labels_e15_1),
                 (2, base_labels_e15_2)]
        for c, reflabels in cases:
            labels = np.zeros(len(neighbourhoods), dtype=np.int_)
            consider = np.ones_like(labels, dtype=np.uint8)
            cfits.fit_from_NeighbourhoodsList(
                neighbourhoods, labels, consider, c
                )

            np.testing.assert_array_equal(
                np.array(reflabels),
                labels
                )

    def test_fit_from_NeighbourhoodsArray_base_neighbourhoods_e15(
            self,
            base_neighbourhoods_e15,
            base_labels_e15_0,
            base_labels_e15_1,
            base_labels_e15_2):
        neighbourhoods = np.array([np.asarray(x, dtype=np.int_)
                                   for x in base_neighbourhoods_e15])
        cases = [(0, base_labels_e15_0),
                 (1, base_labels_e15_1),
                 (2, base_labels_e15_2)]
        for c, reflabels in cases:
            labels = np.zeros(len(neighbourhoods), dtype=np.int_)
            consider = np.ones_like(labels, dtype=np.uint8)
            cfits.fit_from_NeighbourhoodsArray(
                neighbourhoods, labels, consider, c
                )

            np.testing.assert_array_equal(
                np.array(reflabels),
                labels
                )
