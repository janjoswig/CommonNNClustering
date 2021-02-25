import numpy as np

from cnnclustering._types import (
    InputData, NeighboursGetter, Metric, Neighbours
    )
from cnnclustering._fit import FitterDeque


class TestFit:

    def test_fit(self, mocker):
        input_data = mocker.Mock(InputData)
        neighbours_getter = mocker.Mock(NeighboursGetter)
        metric = mocker.Mock(Metric)
        neighbours = mocker.Mock(Neighbours)

        type(input_data).n_points = mocker.PropertyMock(return_value=5)
        type(neighbours).n_points = mocker.PropertyMock(return_value=5)

        input_data.get_neighbours = mocker.Mock(
            return_value=neighbours
            )

        labels = np.zeros(5)
        consider = np.ones_like(labels)
