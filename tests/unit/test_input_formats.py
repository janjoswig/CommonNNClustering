import numpy as np
import pytest

import cnnclustering.cnn as cnn


class TestPoints:
    @pytest.mark.parametrize(
        "points,pshape,edges,eshape ",
        [(None, [1, 0], None, [0]),
         ([1], [1, 1], None, [0]),
         ([1, 2], [1, 2], None, [0]),
         ([[1, 1], [2, 2]], [2, 2], None, [0]),
         pytest.param(
             [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], [2, 2], None, [0],
             marks=pytest.mark.xfail(reason=AssertionError)
             )]
         )
    def test_default_constructor(self, points, pshape, edges, eshape):
        points = cnn.Points(points, edges)
        np.testing.assert_array_equal(points.shape, pshape)
        np.testing.assert_array_equal(points.edges.shape, eshape)
