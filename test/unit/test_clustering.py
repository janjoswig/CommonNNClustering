from cnnclustering import cluster
from cnnclustering._types import (
    InputData,
    NeighboursGetter,
    Metric,
    SimilarityChecker,
    Queue
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
        clustering.fit(radius_cutoff=1., cnn_cutoff=1)

        fitter.fit.assert_called_once()
