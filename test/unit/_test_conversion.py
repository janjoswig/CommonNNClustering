import numpy as np
import pytest

import cnnclustering._cfits as cfits


class TestNeighbouhoods2Graph:
    """Test neighbourhoods to graph conversion"""

    @pytest.mark.skip(reason="Unpredictable outcome (unordered sets)")
    def test_NeighbourhoodsList2SparsegraphArray(self):
        neighbourhoods = [
            {1, 2, 3, 11},  # 0
            {0, 2, 4, 12},  # 1
            {0, 1, 3},  # 2
            {0, 2},  # 3
            {1},  # 4
            {6},  # 5
            {5},  # 6
            set(),  # 7
            {9, 10},  # 8
            {8, 10},  # 9
            {8, 9},  # 10
            {0},  # 11
            {1},  # 12
        ]
        edges, indices = cfits.NeighbourhoodsList2SparsegraphArray(neighbourhoods, 1)
        np.testing.assert_array_equal(
            np.asarray(edges),
            np.array(
                [
                    1,
                    2,
                    3,  # 0
                    0,
                    2,  # 1
                    0,
                    1,
                    3,  # 2
                    0,
                    2,  # 3
                    9,
                    10,  # 8
                    8,
                    10,  # 9
                    8,
                    9,  # 10
                ]
            ),
        )
        np.testing.assert_array_equal(
            indices, np.array([0, 3, 5, 8, 10, 10, 10, 10, 10, 12, 14, 16, 16, 16])
        )
