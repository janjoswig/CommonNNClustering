from collections import Counter

import numpy as np
import pytest

from cnnclustering import cluster
from cnnclustering._primitive_types import P_AINDEX, P_AVALUE
from cnnclustering import _types
from cnnclustering import _fit


class TestClustering:
    def test_create(self):
        clustering = cluster.Clustering(recipe={})
        assert clustering
        assert clustering._bundle is None
        assert clustering._fitter is None
        assert clustering._hierarchical_fitter is None
        assert clustering._predictor is None

    def test_fit_fully_mocked(self, mocker):
        input_data = mocker.Mock(_types.InputData)
        fitter = mocker.Mock(_fit.Fitter)
        type(fitter).make_parameters = mocker.MagicMock(
            return_value=_types.ClusterParameters(1)
            )

        type(input_data).n_points = mocker.PropertyMock(return_value=5)

        clustering = cluster.Clustering(
            data=input_data,
            fitter=fitter,
        )
        clustering.fit(radius_cutoff=1.0, cnn_cutoff=1)

        fitter.fit.assert_called_once()

    def test_predict_fully_mocked(self, mocker):
        input_data = mocker.Mock(_types.InputData)
        predictor = mocker.Mock(_fit.Predictor)

        type(input_data).n_points = mocker.PropertyMock(return_value=5)
        type(predictor).make_parameters = mocker.MagicMock(
            return_value=_types.ClusterParameters(1)
            )

        clustering = cluster.Clustering(
            data=input_data,
            predictor=predictor,
        )

        other_clustering = cluster.Clustering(
            data=input_data,
        )

        clustering.predict(
            other_clustering._bundle,
            radius_cutoff=1.0,
            cnn_cutoff=1,
            clusters={1, 2, 3}
            )

        predictor.predict.assert_called_once()

    @pytest.mark.parametrize(
        "input_data_type,data,meta,labels,root_indices,parent_indices",
        [
            (
                _types.InputDataExtComponentsMemoryview,
                np.array(
                    [[0, 0, 0],
                     [1, 1, 1]],
                    order="C", dtype=P_AVALUE
                ),
                None,
                np.array([1, 2], dtype=P_AINDEX),
                None,
                None
            ),
            (
                _types.InputDataExtComponentsMemoryview,
                np.array(
                    [[0, 0, 0],
                     [1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]],
                    order="C", dtype=P_AVALUE
                ),
                {"edges": [2, 2]},
                np.array([1, 2, 1, 2], dtype=P_AINDEX),
                None,
                None
            ),
            (
                _types.InputDataExtComponentsMemoryview,
                np.array(
                    [[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3],
                     [4, 4, 4]],
                    order="C", dtype=P_AVALUE
                ),
                {"edges": [4]},
                np.array([1, 2, 1, 2], dtype=P_AINDEX),
                np.array([1, 2, 3, 4], dtype=P_AINDEX),
                np.array([1, 2, 3, 4], dtype=P_AINDEX),
            ),
            (
                _types.InputDataExtComponentsMemoryview,
                np.array(
                    [[1, 1, 1],
                     [4, 4, 4]],
                    order="C", dtype=P_AVALUE
                ),
                {"edges": [4]},
                np.array([1, 2], dtype=P_AINDEX),
                np.array([1, 3], dtype=P_AINDEX),
                np.array([0, 2], dtype=P_AINDEX),
            ),
        ]
    )
    def test_isolate(
            self, input_data_type, data, meta, labels,
            root_indices, parent_indices,
            file_regression):

        clustering = cluster.Clustering(
            data=input_data_type(data, meta=meta),
            bundle_kwargs={
                "labels": _types.Labels(labels)
            }
        )
        if not ((root_indices is None) or (parent_indices is None)):
            clustering._bundle._reference_indices = _types.ReferenceIndices(
                root_indices,
                parent_indices
            )

        clustering.isolate()
        label_set = set(labels)
        label_counter = Counter(labels)

        assert len(clustering._bundle._children) == len(label_set)

        report = ""
        for label in label_set:
            isolated_points = clustering._bundle._children[label]._input_data
            assert isolated_points.n_points == label_counter[label]

            edges = isolated_points.meta.get('edges', "None")
            report += (
                f"Child {label}\n"
                f'{"=" * 80}\n'
                f"Data:\n{isolated_points.data}\n"
                f"Edges:\n{edges}\n"
                f"Root:\n{clustering._bundle._children[label].root_indices}\n"
                f"Parent:\n{clustering._bundle._children[label].parent_indices}\n"
                f"\n"
            )

        file_regression.check(report)

    @pytest.mark.parametrize(
        "case_key,depth,expected",
        [
            (
                "hierarchical_a", 1,
                [0, 0, 0, 3, 0, 0, 0, 2, 4, 4, 4, 2, 2, 3, 0]
            ),
            (
                "hierarchical_a", None,
                [0, 0, 0, 3, 0, 0, 0, 2, 6, 5, 0, 2, 2, 3, 0]
            )
        ]
    )
    def test_reel(
            self, case_key, registered_clustering, depth, expected):
        registered_clustering.reel(depth=depth)
        np.testing.assert_array_equal(
            registered_clustering._bundle.labels,
            expected
        )
