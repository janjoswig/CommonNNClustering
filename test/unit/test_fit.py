import numpy as np

from cnnclustering import _fit, _types


class TestFit:

    def test_fit(self, mocker):
        input_data_mock = mocker.Mock(spec=_types.InputData) 
        type(input_data_mock).n_points = mocker.PropertyMock(return_value=5)

        neighbours_mock = mocker.Mock(spec=_types.Neighbours)
        type(neighbours_mock).n_points = mocker.PropertyMock(return_value=5)

        input_data_mock.get_neighbours = mocker.Mock(
            return_value=neighbours_mock
            )

        labels = np.zeros(5)
        consider = np.ones_like(labels)

        _fit.fit_deque(input_data_mock, labels, consider)

        print(labels, consider)
