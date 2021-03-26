import pytest

from cnnclustering import plot


@pytest.mark.parametrize(
    "case_key",
    ["hierarchical_a"]
)
def test_get_pieces(case_key, registered_clustering):
    pieces = plot.get_pieces(registered_clustering)
    assert len(set([sum(x[1] for x in y) for y in pieces])) == 1
