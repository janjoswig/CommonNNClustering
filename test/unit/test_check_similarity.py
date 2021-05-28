import pytest

from cnnclustering._types import (
    NeighboursExtVector,
    NeighboursList,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    SimilarityCheckerSwitchContains,
    SimilarityCheckerExtSwitchContains,
    SimilarityCheckerExtScreensorted
)
from cnnclustering._types import ClusterParameters


class TestSimilarityChecker:
    @pytest.mark.parametrize(
        "checker_type,checker_is_ext,needs_sorted",
        [
            (SimilarityCheckerContains, False, False),
            (SimilarityCheckerSwitchContains, False, False),
            (SimilarityCheckerExtContains, True, False),
            (SimilarityCheckerExtSwitchContains, True, False),
            (SimilarityCheckerExtScreensorted, True, True)
        ]
    )
    @pytest.mark.parametrize(
        "neighbours_type,args,kwargs,neighbours_is_ext",
        [
            (NeighboursList, (), {}, False),
            (NeighboursExtVector, (10,), {}, True),
        ],
    )
    @pytest.mark.parametrize(
        "members_a,members_b,c,is_sorted,expected",
        [
            ([], [], 0, True, True),
            ([], [], 1, True, False),
            ([1, 2, 3], [2, 5], 1, True, True),
            ([1, 2, 3], [2, 5, 8, 9], 2, True, False),
            ([1, 2, 3, 9, 10], [2, 5, 8, 9, 10], 3, True, True),
            ([3, 2, 1, 18, 9, 10], [2, 5, 9, 8, 10, 11, 18], 3, False, True),
        ]
    )
    def test_check(
            self,
            checker_type, checker_is_ext, needs_sorted,
            neighbours_type, args, kwargs, neighbours_is_ext,
            members_a, members_b, c, is_sorted, expected):

        if checker_is_ext and (not neighbours_is_ext):
            # pytest.skip("Bad combination of component types.")
            return

        if not is_sorted == needs_sorted:
            return

        neighbours_a = neighbours_type(*args, **kwargs)
        neighbours_b = neighbours_type(*args, **kwargs)
        for member in members_a:
            neighbours_a.assign(member)
        for member in members_b:
            neighbours_b.assign(member)

        cluster_params = ClusterParameters(radius_cutoff=0.0, cnn_cutoff=c)

        checker = checker_type()
        passed = checker.check(neighbours_a, neighbours_b, cluster_params)
        assert passed == expected
