import pytest

from cnnclustering import plot


@pytest.mark.parametrize(
    "case_key",
    ["hierarchical_a"]
)
def test_getpieces(case_key, registered_clustering):
    pieces = plot.getpieces(registered_clustering)
    for ref_shares_mapping in pieces.values():
        shares_sum = sum(
            sum(shares.values())
            for shares in ref_shares_mapping.values()
        )
        # assert shares_sum == 1
        print(shares_sum)
        print(registered_clustering.labels.shape[0])
