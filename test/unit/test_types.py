import numpy as np
import pytest

from cnnclustering._primitive_types import P_AVALUE, P_AINDEX, P_ABOOL
from cnnclustering._types import (
    ClusterParameters,
    InputDataExtPointsMemoryview,
    InputDataNeighboursSequence,
    Labels,
    NeighboursSet,
    NeighboursList,
    NeighboursExtVector,
    NeighboursGetterLookup,
    SimilarityCheckerExtScreensorted,
    QueueFIFODeque,
    QueueExtFIFOQueue,
    QueueExtLIFOVector,
)


class TestClusterParameters:
    @pytest.mark.parametrize("radius_cutoff,cnn_cutoff", [(0.1, 1)])
    def test_create_params(self, radius_cutoff, cnn_cutoff, file_regression):
        cluster_params = ClusterParameters(radius_cutoff, cnn_cutoff)
        repr_ = f"{cluster_params!r}"
        str_ = f"{cluster_params!s}"
        file_regression.check(f"{repr_}\n{str_}")


class TestLabels:
    @pytest.mark.parametrize(
        "labels,consider,meta,constructor",
        [
            (np.zeros(10, dtype=P_AINDEX), None, None, None),
            (
                np.zeros(10, dtype=P_AINDEX),
                np.ones(10, dtype=P_ABOOL),
                None,
                None
            ),
            pytest.param(
                np.zeros(10, dtype=P_AINDEX),
                np.ones(9, dtype=P_ABOOL),
                None,
                None,
                marks=[pytest.mark.raises(exception=ValueError)],
            ),
            (
                [0, 0, 0],
                [1, 1, 1],
                None,
                "from_sequence"
            ),
        ],
    )
    def test_create_labels(
            self, labels, consider, meta, constructor, file_regression):
        if constructor is None:
            _labels = Labels(labels, consider=consider, meta=meta)
        else:
            _labels = getattr(Labels, constructor)(
                labels,
                consider=consider,
                meta=meta
                )

        assert isinstance(_labels.labels, np.ndarray)
        assert isinstance(_labels.consider, np.ndarray)
        assert isinstance(_labels.meta, dict)

        repr_ = f"{_labels!r}"
        str_ = f"{_labels!s}"
        file_regression.check(f"{repr_}\n{str_}")

    def test_attribute_access(self):
        _labels = Labels.from_sequence([])

        assert _labels.set == set()
        assert _labels.mapping == {}

        assert _labels.consider_set == set()
        _labels.consider_set = {1}
        assert _labels.consider_set == {1}

    def test_to_mapping(self):
        labels = Labels(np.array([1, 1, 2, 2, 0, 1], dtype=P_AINDEX, order="C"))
        mapping = labels.to_mapping()
        assert mapping == labels.mapping
        assert mapping == {0: [4], 1: [0, 1, 5], 2: [2, 3]}

    def test_to_set(self):
        labels = Labels(np.array([1, 1, 2, 2, 0, 1], dtype=P_AINDEX, order="C"))
        label_set = labels.to_set()
        assert label_set == labels.set
        assert label_set == {0, 1, 2}

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"member_cutoff": 1}, [1, 1, 3, 3, 0, 1, 2, 2, 2, 1, 4]),
            ({}, [1, 1, 3, 3, 0, 1, 2, 2, 2, 1, 0]),
            (
                {"member_cutoff": 2, "max_clusters": 2},
                [1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 0]
            ),
        ]
    )
    def test_sort_by_size(self, kwargs, expected):
        labels = Labels(
            np.array([1, 1, 2, 2, 0, 1, 3, 3, 3, 1, 5], dtype=P_AINDEX, order="C")
            )
        labels.sort_by_size(**kwargs)
        np.testing.assert_array_equal(
            labels.labels,
            expected
        )


class TestInputData:
    @pytest.mark.parametrize(
        (
            "input_data_type,data,n_points,n_dim,n_neighbours,"
            "neighbour_queries,component_queries"
        ),
        [
            (
                InputDataNeighboursSequence,
                [[0, 1], [0, 1]], 2, 0, (2, 2), [(0, 0, 0)], [(1, 1, 0)]

            ),
            (
                InputDataExtPointsMemoryview,
                np.array([[0, 1], [0, 1]], order="C", dtype=P_AVALUE),
                2, 2, (0, 0), [(0, 0, 0)], [(1, 1, 1)]
            ),
        ],
    )
    def test_create_input_data(
            self,
            input_data_type,
            data,
            n_points, n_dim,
            n_neighbours,
            neighbour_queries,
            component_queries):

        input_data = input_data_type(data)

        assert input_data.n_points == n_points
        assert input_data.n_dim == n_dim

        for c, n in enumerate(n_neighbours):
            assert n == input_data.get_n_neighbours(c)

        for point, member, expected in neighbour_queries:
            assert expected == input_data.get_neighbour(point, member)

        for point, dim, expected in component_queries:
            assert expected == input_data.get_component(point, dim)

        assert isinstance(input_data.data, (np.ndarray, list))

    @pytest.mark.parametrize(
        "input_data_type,data,indices,expected",
        [
            (
                InputDataExtPointsMemoryview,
                np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8],
                          [9, 10, 11]], dtype=P_AVALUE, order="C"),
                [1, 2],
                np.array([[3, 4, 5],
                          [6, 7, 8]])
            ),
            (
                InputDataNeighboursSequence,
                [[1, 2, 3], [0, 2, 4], [0, 1, 4], [0], [1, 2]],
                [1, 2],
                [[2], [1]]
            ),
        ]
    )
    def test_get_subset(self, input_data_type, data, indices, expected):
        input_data = input_data_type(data)

        input_data_subset = input_data.get_subset(indices)

        np.testing.assert_array_equal(
            input_data_subset.data,
            expected
        )


class TestNeighbours:
    @pytest.mark.parametrize(
        "neighbours_type,args,kwargs,n_points,",
        [
            (NeighboursList, ([0, 1, 3],), {}, 3),
            (NeighboursSet, ({0, 1, 3},), {}, 3),
            (NeighboursExtVector, (5, [0, 1, 3],), {}, 3),
        ]
    )
    def test_neighbours(self, neighbours_type, args, kwargs, n_points):
        neighbours = neighbours_type(*args, **kwargs)
        assert neighbours.n_points == n_points

        neighbours.assign(5)
        assert neighbours.n_points == n_points + 1
        assert neighbours.get_member(n_points) == 5
        assert neighbours.contains(5)
        assert not neighbours.contains(99)
        assert neighbours.enough(3)
        assert not neighbours.enough(4)

        neighbours.reset()
        assert neighbours.n_points == 0


class TestNeighboursGetter:
    def test_get(self):
        input_data = InputDataNeighboursSequence([[1, 2, 3], [0, 2, 3, 4]])
        neighbours = NeighboursList()
        neighbours_getter = NeighboursGetterLookup()
        cluster_params = ClusterParameters(
            radius_cutoff=0, cnn_cutoff=0
            )

        neighbours_getter.get(0, input_data, neighbours, None, cluster_params)
        assert neighbours._neighbours == [1, 2, 3]


class TestQueue:

    @pytest.mark.parametrize(
        "queue_type,kind",
        [
            (QueueFIFODeque, "fifo"),
            (QueueExtFIFOQueue, "fifo"),
            (QueueExtLIFOVector, "lifo"),
        ]
    )
    def test_use_queue(
            self, queue_type, kind):
        queue = queue_type()

        assert queue.is_empty()

        queue.push(1)
        assert queue.pop() == 1

        pushed = list(range(10))
        for i in pushed:
            queue.push(i)

        popped = []
        while not queue.is_empty():
            popped.append(queue.pop())

        if kind == "lifo":
            popped.reverse()
        assert pushed == popped
