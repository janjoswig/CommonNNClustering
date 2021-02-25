import pytest

from cnnclustering._types import InputDataNeighboursSequence
from cnnclustering._types import NeighboursSequence


class TestInputData:

    @pytest.mark.parametrize(
        "input_data_type,data,expected",
        [
            (InputDataNeighboursSequence, [[0, 1], [0, 1]], 2)
        ]
    )
    def test_n_points(self, input_data_type, data, expected):
        input_data = input_data_type(data)
        assert input_data.n_points == expected


class TestNeighbours:

    @pytest.mark.parametrize(
        "neighbours_type,data,expected",
        [
            (NeighboursSequence, [0, 1], 2)
        ]
    )
    def test_n_points(self, neighbours_type, data, expected):
        neighbours = neighbours_type(data)
        assert neighbours.n_points == expected
