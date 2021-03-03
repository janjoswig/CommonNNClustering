import numpy as np
import pytest

from cnnclustering._primitive_types import P_AINDEX
from cnnclustering._types import (
    NeighboursExtMemoryview,
    NeighboursSequence,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    SimilarityCheckerSwitchContains,
    SimilarityCheckerExtSwitchContains,
)
from cnnclustering._types import ClusterParameters


class TestSimilarityChecker:

    @pytest.mark.parametrize(
        "checker",
        [SimilarityCheckerContains, SimilarityCheckerSwitchContains]
    )
    @pytest.mark.parametrize(
        "neighbours_type,data_a,data_b,c,expected",
        [
            (NeighboursSequence, [], [], 0, True),
            (NeighboursSequence, [], [], 1, False),
            (NeighboursSequence, [1, 2, 3], [2, 5], 1, True),
            (NeighboursSequence, [1, 2, 3], [2, 5, 9, 8], 2, False),
        ],
    )
    def test_check(
            self, checker, neighbours_type, data_a, data_b, c, expected):
        neighbours_a = neighbours_type(data_a)
        neighbours_b = neighbours_type(data_b)
        cluster_params = ClusterParameters(radius_cutoff=0., cnn_cutoff=c)

        passed = checker().check(neighbours_a, neighbours_b, cluster_params)
        assert passed == expected

    @pytest.mark.parametrize(
        "checker",
        [SimilarityCheckerExtContains, SimilarityCheckerExtSwitchContains]
    )
    @pytest.mark.parametrize(
        "neighbours_type,data_a,data_b,c,expected",
        [
            (
                NeighboursExtMemoryview,
                np.array([], dtype=P_AINDEX),
                np.array([], dtype=P_AINDEX), 0, True
            ),
            (
                NeighboursExtMemoryview,
                np.array([], dtype=P_AINDEX),
                np.array([], dtype=P_AINDEX), 1, False
            ),
            (
                NeighboursExtMemoryview,
                np.array([1, 2, 3], dtype=P_AINDEX),
                np.array([2, 5], dtype=P_AINDEX), 1, True
            ),
            (
                NeighboursExtMemoryview,
                np.array([1, 2, 3], dtype=P_AINDEX),
                np.array([2, 5, 9, 8], dtype=P_AINDEX), 2, False
            ),
        ],
    )
    def test_check_ext(
            self, checker, neighbours_type, data_a, data_b, c, expected):
        neighbours_a = neighbours_type(data_a)
        neighbours_b = neighbours_type(data_b)
        cluster_params = ClusterParameters(radius_cutoff=0., cnn_cutoff=c)

        passed = checker()._check(neighbours_a, neighbours_b, cluster_params)
        assert passed == expected
