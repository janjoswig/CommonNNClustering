import numpy as np
import pytest

from cnnclustering import cluster
from cnnclustering._types import (
    InputData,
    NeighboursGetter,
    Metric,
    SimilarityChecker,
    Queue,
)
from cnnclustering._fit import Fitter


class TestClustering:
    def test_create(self):
        clustering = cluster.Clustering()
        assert clustering

    def test_fit_fully_mocked(self, mocker):
        input_data = mocker.Mock(InputData)
        neighbours_getter = mocker.Mock(NeighboursGetter)
        similarity_checker = mocker.Mock(SimilarityChecker)
        metric = mocker.Mock(Metric)
        queue = mocker.Mock(Queue)
        fitter = mocker.Mock(Fitter)

        type(input_data).n_points = mocker.PropertyMock(return_value=5)

        clustering = cluster.Clustering(
            input_data=input_data,
            neighbours_getter=neighbours_getter,
            metric=metric,
            similarity_checker=similarity_checker,
            queue=queue,
            fitter=fitter,
        )
        clustering.fit(radius_cutoff=1.0, cnn_cutoff=1)

        fitter.fit.assert_called_once()


class TestPreparationHooks:

    @pytest.mark.parametrize(
        "data,expected_data,expected_meta",
        [
            pytest.param(
                1, None, None,
                marks=pytest.mark.raises(exception=TypeError)
            ),
            ([], [[]], {"edges": [1]}),
            ([1, 2, 3], [[1, 2, 3]], {"edges": [1]}),
            pytest.param(
                [[1, 2, 3], [4, 5]], None, None,
                marks=pytest.mark.raises(exception=ValueError)
            ),
            (
                [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                {"edges": [2]}
            ),
            pytest.param(
                [[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11], [13, 14, 15]]], None, None,
                marks=pytest.mark.raises(exception=ValueError)
            ),
            (
                [[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11, 12], [13, 14, 15]]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
                {"edges": [2, 3]}
            ),
        ],
        ids=[
            "invalid", "empty", "1d", "2d_invalid", "2d", "1d2d_invalid",
            "1d2d"
            ]
    )
    def test_prepare_points_from_parts(
            self, data, expected_data, expected_meta):
        reformatted_data, meta = cluster.prepare_points_from_parts(data)
        print(reformatted_data)
        np.testing.assert_array_equal(
            expected_data,
            reformatted_data
            )
        assert meta == expected_meta
