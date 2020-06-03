"""Testing functions calculating data point neighbourhoods"""

import numpy as np

import cnnclustering._cfits as cfits


class TestPython:
    pass


class TestCython:
    def test_get_neighbours_PointsArray_base_points_e15(
            self,
            base_points,
            base_neighbourhoods_e15):
        points = np.array(base_points, dtype=np.float_)
        n = points.shape[0]
        for i in range(n):
            neighbours = cfits._get_neighbours_PointsArray(
                i, points, 1.5**2
                )
            assert set(base_neighbourhoods_e15[i]) == neighbours
