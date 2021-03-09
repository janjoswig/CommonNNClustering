import pytest

from cnnclustering._types import (
    NeighboursExtVector,
    NeighboursList,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    SimilarityCheckerSwitchContains,
    SimilarityCheckerExtSwitchContains,
)
from cnnclustering._types import ClusterParameters


class TestSimilarityChecker:
    @pytest.mark.parametrize(
        "checker_type,checker_is_ext",
        [
            (SimilarityCheckerContains, False),
            (SimilarityCheckerSwitchContains, False),
            (SimilarityCheckerExtContains, True),
            (SimilarityCheckerExtSwitchContains, True)
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
        "members_a,members_b,c,expected",
        [
            ([], [], 0, True),
            ([], [], 1, False),
            ([1, 2, 3], [2, 5], 1, True),
            ([1, 2, 3], [2, 5, 9, 8], 2, False),
            ([1, 2, 3, 9, 10], [2, 5, 9, 8, 10], 3, True),
        ]
    )
    def test_check(
            self,
            checker_type, checker_is_ext,
            neighbours_type, args, kwargs, neighbours_is_ext,
            members_a, members_b, c, expected):
        all_ext = (checker_is_ext is True and neighbours_is_ext is True)
        all_object = (checker_is_ext is False and neighbours_is_ext is False)
        if not (all_ext or all_object):
            # pytest.skip("Bad combination of component types.")
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
