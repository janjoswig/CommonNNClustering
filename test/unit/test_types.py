import numpy as np
import pytest

from cnnclustering._primitive_types import P_AVALUE, P_AINDEX, P_ABOOL
from cnnclustering._types import (
    ClusterParameters,
    InputDataExtPointsMemoryview,
    InputDataNeighboursSequence,
    Labels,
    NeighboursSequence
    )


class TestClusterParameters:

    @pytest.mark.parametrize(
        "radius_cutoff,cnn_cutoff",
        [(0.1, 1)]
    )
    def test_create_params(self, radius_cutoff, cnn_cutoff, file_regression):
        cluster_params = ClusterParameters(radius_cutoff, cnn_cutoff)
        repr_ = f"{cluster_params!r}"
        str_ = f"{cluster_params!s}"
        file_regression.check(f"{repr_}\n{str_}")


class TestLabels:

    @pytest.mark.parametrize(
        "labels,consider",
        [
            (np.zeros(10, dtype=P_AINDEX), None),
            (np.zeros(10, dtype=P_AINDEX), np.ones(10, dtype=P_ABOOL)),
            pytest.param(
                np.zeros(10, dtype=P_AINDEX),
                np.ones(9, dtype=P_ABOOL),
                marks=[pytest.mark.raises(exception=ValueError)]
                ),
        ]
    )
    def test_create_labels(self, labels, consider, file_regression):
        _labels = Labels(labels, consider)

        assert isinstance(_labels.labels, np.ndarray)
        assert isinstance(_labels.consider, np.ndarray)

        repr_ = f"{_labels!r}"
        str_ = f"{_labels!s}"
        file_regression.check(f"{repr_}\n{str_}")


class TestInputData:

    @pytest.mark.parametrize(
        "input_data_type,data,expected",
        [
            (InputDataNeighboursSequence, [[0, 1], [0, 1]], 2),
            (
                InputDataExtPointsMemoryview,
                np.array([[0, 1], [0, 1]], order="c", dtype=P_AVALUE),
                2
            )
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
