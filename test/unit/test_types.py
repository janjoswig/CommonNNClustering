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
        "labels,consider",
        [
            (np.zeros(10, dtype=P_AINDEX), None),
            (np.zeros(10, dtype=P_AINDEX), np.ones(10, dtype=P_ABOOL)),
            pytest.param(
                np.zeros(10, dtype=P_AINDEX),
                np.ones(9, dtype=P_ABOOL),
                marks=[pytest.mark.raises(exception=ValueError)],
            ),
        ],
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
        "input_data_type,data,n_points,n_dim,n_neighbours,neighbour_queries,component_queries",
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
    def test_n_points(
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
        assert neighbours.enough(3)
        assert not neighbours.enough(4)


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
