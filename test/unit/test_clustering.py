from collections import Counter

import numpy as np
import pytest

from cnnclustering import cluster, hooks
from cnnclustering._primitive_types import P_AINDEX, P_AVALUE
from cnnclustering import _types
from cnnclustering import _fit


class TestClustering:
    def test_create(self):
        clustering = cluster.Clustering()
        assert clustering

    def test_fit_fully_mocked(self, mocker):
        input_data = mocker.Mock(_types.InputData)
        fitter = mocker.Mock(_fit.Fitter)

        type(input_data).n_points = mocker.PropertyMock(return_value=5)

        clustering = cluster.Clustering(
            input_data=input_data,
            fitter=fitter,
        )
        clustering.fit(radius_cutoff=1.0, cnn_cutoff=1)

        fitter.fit.assert_called_once()

    def test_predict_fully_mocked(self, mocker):
        input_data = mocker.Mock(_types.InputData)
        predictor = mocker.Mock(_fit.Predictor)

        type(input_data).n_points = mocker.PropertyMock(return_value=5)

        clustering = cluster.Clustering(
            input_data=input_data,
            predictor=predictor,
        )

        other_clustering = cluster.Clustering(
            input_data=input_data,
        )

        clustering.predict(
            other_clustering,
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
            input_data=input_data_type(data, meta=meta),
            labels=_types.Labels(labels)
        )
        clustering._root_indices = root_indices
        clustering._parent_indices = parent_indices
        clustering.isolate()
        label_set = set(labels)
        label_counter = Counter(labels)

        assert len(clustering._children) == len(label_set)

        report = ""
        for label in label_set:
            isolated_points = clustering._children[label]._input_data
            assert isolated_points.n_points == label_counter[label]

            edges = isolated_points.meta.get('edges', "None")
            report += (
                f"Child {label}\n"
                f'{"=" * 80}\n'
                f"Data:\n{isolated_points.data}\n"
                f"Edges:\n{edges}\n"
                f"Root:\n{clustering._children[label]._root_indices}\n"
                f"Parent:\n{clustering._children[label]._parent_indices}\n"
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
            registered_clustering._labels.labels,
            expected
        )


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
        data_args, data_kwargs = hooks.prepare_points_from_parts(data)
        reformatted_data = data_args[0]
        np.testing.assert_array_equal(
            expected_data,
            reformatted_data
            )
        assert data_kwargs["meta"] == expected_meta
