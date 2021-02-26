import pytest

from cnnclustering._types import (
    SimilarityCheckerContains,
    SimilarityCheckerSwitchContains,
    SimilarityCheckerExtContains,
    SimilarityCheckerExtSwitchContains,
    NeighboursSequence,
    NeighboursExtMemoryview
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
