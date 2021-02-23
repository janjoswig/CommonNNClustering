import numpy as np
import pytest

import cnnclustering.cnn as cnn


TESTCASES = (
    "points_passed,expected_pshape,edges_passed,expected_edges_deduced",
    [
        (None, [1, 0], None, []),
        ([1], [1, 1], None, [1]),
        ([1, 1], [1, 2], None, [1]),
        ([[1, 1], [2, 2]], [2, 2], None, [2]),
        ([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], [4, 2], None, [2, 2]),
        ([[1, 1], [2, 2], [3, 3], [4, 4]], [4, 2], [2, 2], [4])
        ]
    )


class TestPoints:
    @pytest.mark.parametrize(
        *TESTCASES,
        ids=["None", "1_1d", "2_1d", "2_2d", "2p_2_2d_fail", "4_2d_e2p"]
        )
    def test_default_constructor(
            self,
            points_passed, expected_pshape,
            edges_passed, expected_edges_deduced):

        if edges_passed is None:
            expected_edges_deduced = []
        else:
            expected_edges_deduced = edges_passed

        points = cnn.Points(points_passed, edges_passed)
        np.testing.assert_array_equal(points.shape, expected_pshape)
        np.testing.assert_array_equal(points.edges, expected_edges_deduced)

    @pytest.mark.parametrize(
        *TESTCASES,
        ids=["None", "1_1d", "2_1d", "2_2d", "2p_2_2d", "4_2d_e2p"]
        )
    def test_from_parts(
            self,
            points_passed, expected_pshape,
            edges_passed, expected_edges_deduced):

        points = cnn.Points.from_parts(points_passed)
        np.testing.assert_array_equal(points.shape, expected_pshape)
        np.testing.assert_array_equal(points.edges, expected_edges_deduced)

    @pytest.mark.parametrize(
        *TESTCASES,
        ids=["None", "1_1d", "2_1d", "2_2d", "2p_2_2d", "4_2d_e2p"]
        )
    def test_by_parts(
            self,
            points_passed, expected_pshape,
            edges_passed, expected_edges_deduced):

        points = cnn.Points.from_parts(points_passed)
        points_by_parts = list(points.by_parts())
        if points.size == 0:
            # yield from ()
            assert len(points_by_parts) == 0
        else:
            if points.edges.size == 0:
                # Assume one part
                assert len(points_by_parts) == 1
            else:
                # Parts have correct number of points?
                for c, part_members in enumerate(points.edges):
                    len(points_by_parts[c]) == part_members
            # Yielded points match original?
            np.testing.assert_array_equal(points, np.vstack(points_by_parts))
