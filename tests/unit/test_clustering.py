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
                neighbourhoods, labels, consider, c, False
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
                neighbourhoods, labels, consider, c, False
                )

            np.testing.assert_array_equal(
                np.array(reflabels),
                labels
                )

    def test_fit_from_SparsegraphArray_base_densitygraphs_e15(
            self,
            base_densitygraph_e15_0,
            base_densitygraph_e15_1,
            base_densitygraph_e15_2,
            base_labels_e15_0,
            base_labels_e15_1,
            base_labels_e15_2):

        cases = [(base_densitygraph_e15_0, base_labels_e15_0),
                 (base_densitygraph_e15_1, base_labels_e15_1),
                 (base_densitygraph_e15_2, base_labels_e15_2)]

        for graph, reflabels in cases:
            vertices = np.asarray(graph[0], dtype=cfits.ARRAYINDEX_DTYPE)
            indices = np.asarray(graph[1], dtype=cfits.ARRAYINDEX_DTYPE)
            labels = np.zeros(indices.shape[0] - 1, dtype=np.int_)
            consider = np.ones_like(labels, dtype=np.uint8)
            cfits.fit_from_SparsegraphArray(
                vertices, indices, labels, consider
                )

            np.testing.assert_array_equal(
                np.array(reflabels),
                labels
                )
