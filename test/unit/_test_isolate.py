from collections import defaultdict, Counter

import numpy as np
import pytest

import cnnclustering.cnn as cnn


TESTCASES = [
    "points,labels", [
        ([[1], [2], [3], [4]], [1, 1, 2, 2]),
        ([[[1], [2]], [[3], [4]]], [1, 2, 1, 2])
    ]
]


class TestIsolate:
    def test_isolate_empty(self, empty_cobj):
        empty_cobj.isolate()
        assert isinstance(empty_cobj.children, defaultdict)
        assert len(empty_cobj.children) == 0

    @pytest.mark.parametrize(*TESTCASES)
    def test_isolate_1p(self, points, labels):
        cobj = cnn.CNN(points=points)
        cobj.labels = cnn.Labels(labels)
        label_set = set(labels)
        label_counter = Counter(labels)
        cobj.isolate()
        assert len(cobj.children) == len(label_set)
        for label in label_set:
            isolated_points = cobj.children[label].data.points.shape[0]
            assert isolated_points == label_counter[label]

    def test_isolate_2p(self, empty_cobj):
        empty_cobj.data.points = cnn.Points.from_parts([[[1], [2]], [[3], [4]]])
        empty_cobj.labels = cnn.Labels([1, 2, 1, 2])
        empty_cobj.isolate()
        assert len(empty_cobj.children) == 2
        assert (
            empty_cobj.children[1].data.points.shape[0] ==
            empty_cobj.children[2].data.points.shape[0]
            )

        np.testing.assert_array_equal(
            empty_cobj.children[1].data.points.edges,
            empty_cobj.children[2].data.points.edges
            )
        np.testing.assert_array_equal(
            empty_cobj.children[1]._refindex, [0, 2]
            )
        np.testing.assert_array_equal(
            empty_cobj.children[2]._refindex, [1, 3]
            )
