"""Testing functions calculating data point neighbourhoods"""

import numpy as np
import pytest

import cnnclustering._cfits as cfits


class TestPython:
    pass


class TestCython:
    @pytest.mark.parametrize("radius_cutoff", [1.5])
    def test_get_neighbours_PointsArray_base_points(
            self,
            radius_cutoff, base_data):
        points = np.array(base_data["points"], dtype=np.float_)
        n = points.shape[0]
        for i in range(n):
            neighbours = cfits._get_neighbours_PointsArray(
                i, points, radius_cutoff**2
                )
            assert set(base_data["neighbourhoods"][i]) == neighbours
