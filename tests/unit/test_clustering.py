import numpy as np
import pytest

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

    @pytest.mark.parametrize("radius_cutoff", [1.5])
    @pytest.mark.parametrize("cnn_cutoff", [0, 1, 2])
    def test_fit_from_PointsArray_base_points(
            self,
            radius_cutoff, cnn_cutoff, base_data):
        points = np.array(base_data["points"], dtype=np.float_)
        labels = np.zeros(points.shape[0], dtype=np.int_)
        consider = np.ones_like(labels, dtype=np.uint8)
        cfits.fit_from_PointsArray(
            points, labels, consider, radius_cutoff, cnn_cutoff
            )

        np.testing.assert_array_equal(
            np.array(base_data["labels"]),
            labels
            )

    @pytest.mark.parametrize("radius_cutoff", [1.5])
    @pytest.mark.parametrize("cnn_cutoff", [0, 1, 2])
    def test_fit_from_DistancesArray_base_points(
            self,
            radius_cutoff, cnn_cutoff, base_data):
        distances = np.array(base_data["distances"], dtype=np.float_)
        labels = np.zeros(distances.shape[0], dtype=np.int_)
        consider = np.ones_like(labels, dtype=np.uint8)
        cfits.fit_from_DistancesArray(
            distances, labels, consider, radius_cutoff, cnn_cutoff)

        np.testing.assert_array_equal(
            np.array(base_data["labels"]),
            labels
            )

    @pytest.mark.parametrize("radius_cutoff", [1.5])
    @pytest.mark.parametrize("cnn_cutoff", [0, 1, 2])
    def test_fit_from_NeighbourhoodsList_base_neighbourhoods(
            self,
            radius_cutoff, cnn_cutoff, base_data):
        neighbourhoods = [set(x) for x in base_data["neighbourhoods"]]

        labels = np.zeros(len(neighbourhoods), dtype=np.int_)
        consider = np.ones_like(labels, dtype=np.uint8)
        cfits.fit_from_NeighbourhoodsList(
            neighbourhoods, labels, consider, cnn_cutoff, False
            )

        np.testing.assert_array_equal(
            np.array(base_data["labels"]),
            labels
            )

    @pytest.mark.parametrize("radius_cutoff", [1.5])
    @pytest.mark.parametrize("cnn_cutoff", [0, 1, 2])
    def test_fit_from_NeighbourhoodsArray_base_neighbourhoods_e15(
            self,
            radius_cutoff, cnn_cutoff, base_data):
        neighbourhoods = np.array([np.asarray(x, dtype=np.int_)
                                   for x in base_data["neighbourhoods"]])

        labels = np.zeros(len(neighbourhoods), dtype=np.int_)
        consider = np.ones_like(labels, dtype=np.uint8)
        cfits.fit_from_NeighbourhoodsArray(
            neighbourhoods, labels, consider, cnn_cutoff, False
            )

        np.testing.assert_array_equal(
            np.array(base_data["labels"]),
            labels
            )

    @pytest.mark.parametrize("radius_cutoff", [1.5])
    @pytest.mark.parametrize("cnn_cutoff", [0, 1, 2])
    def test_fit_from_SparsegraphArray_base_densitygraphs_e15(
            self,
            radius_cutoff, cnn_cutoff, base_data):

        vertices = np.asarray(
            base_data["densitygraph"][0], dtype=cfits.ARRAYINDEX_DTYPE
            )
        indices = np.asarray(
            base_data["densitygraph"][1], dtype=cfits.ARRAYINDEX_DTYPE
            )
        labels = np.zeros(indices.shape[0] - 1, dtype=np.int_)
        consider = np.ones_like(labels, dtype=np.uint8)
        cfits.fit_from_SparsegraphArray(
            vertices, indices, labels, consider
            )

        np.testing.assert_array_equal(
            np.array(base_data["labels"]),
            labels
            )
