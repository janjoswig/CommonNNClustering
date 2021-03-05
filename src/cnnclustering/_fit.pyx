from abc import ABC, abstractmethod
from collections import deque
from typing import Type

from cython.operator cimport dereference as deref

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from cnnclustering._types import (
    InputData,
    NeighboursGetter,
    Neighbours,
    Metric,
    SimilarityChecker,
    Queue,
)

class Fitter(ABC):
    """Defines the fitter interface"""

    @abstractmethod
    def fit(
        self,
        input_data: Type['InputData'],
        neighbours_getter: Type['NeighboursGetter'],
        neighbours: Type['Neighbours'],
        neighbour_neighbours: Type['Neighbours'],
        metric: Type['Metric'],
        similarity_checker: Type['SimilarityChecker'],
        queue: Type['Queue'],
        labels: Type['Labels'],
        cluster_params: Type['ClusterParameters']):
        """Generic clustering"""


class FitterBFS:
    """Concrete implementation of the fitter interface"""

    def fit(
            self,
            object input_data,
            object neighbours_getter,
            object neighbours,
            object neighbour_neighbours,
            object metric,
            object similarity_checker,
            object queue,
            Labels labels,
            ClusterParameters cluster_params):
        """Generic common-nearest-neighbours clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            neighbours_getter: Calculator implementing the
                neighbours-getter interface.
            neighbours: Neighbours container implementing the
                neighbours interface.
            neighbour_neighbours: Neighbours container implementing the
                neighbours interface.
            metric: Distance metric implementing the metric
                interface.
            similarity_checker: Density-criterion evaluator implementing
                the similarity-checker interface.
            queue: Queing data structure implementing the queue
                interface.
            labels: Instance of :obj:`cnnclustering._types.Labels`.
            cluster_params: Instance of
                :obj:`cnnclustering._types.ClusterParameters`.
        """

        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data.n_points
        current = 1

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            neighbours_getter.get(
                init_point,
                input_data,
                neighbours,
                metric,
                cluster_params
                )

            if not neighbours.enough(cluster_params):
                continue

            _labels[init_point] = current

            while True:

                m = neighbours.n_points

                for member_index in range(m):
                    member = neighbours.get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    neighbours_getter.get(
                        member,
                        input_data,
                        neighbour_neighbours,
                        metric,
                        cluster_params
                        )

                    if not neighbour_neighbours.enough(cluster_params):
                        _consider[member] = 0
                        continue

                    if similarity_checker.check(
                            neighbours,
                            neighbour_neighbours,
                            cluster_params):
                        _consider[member] = 0
                        _labels[member] = current
                        queue.push(member)

                if queue.is_empty():
                    break

                point = queue.pop()
                neighbours_getter.get(
                    point,
                    input_data,
                    neighbours,
                    metric,
                    cluster_params
                    )

            current += 1


cdef class FitterExtBFS:
    """Concrete implementation of the fitter interface"""

    cdef void _fit(
            self,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_GETTER_EXT neighbours_getter,
            NEIGHBOURS_EXT neighbours,
            NEIGHBOUR_NEIGHBOURS_EXT neighbour_neighbours,
            METRIC_EXT metric,
            SIMILARITY_CHECKER_EXT similarity_checker,
            QUEUE_EXT queue,
            Labels labels,
            ClusterParameters cluster_params):
        """Generic common-nearest-neighbours clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            neighbours_getter: Calculator implementing the
                neighbours-getter interface.
            neighbours: Neighbours container implementing the
                neighbours interface.
            neighbour_neighbours: Neighbours container implementing the
                neighbours interface.
            metric: Distance metric implementing the metric
                interface.
            similarity_checker: Density-criterion evaluator implementing
                the similarity-checker interface.
            queue: Queing data structure implementing the queue
                interface.
            labels: Instance of :obj:`cnnclustering._types.Labels`.
            cluster_params: Instance of
                :obj:`cnnclustering._types.ClusterParameters`.
        """

        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data.n_points
        current = 1

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            neighbours_getter._get(
                init_point,
                input_data,
                neighbours,
                metric,
                cluster_params
                )

            if not neighbours._enough(cluster_params):
                continue

            _labels[init_point] = current

            while True:

                m = neighbours.n_points

                for member_index in range(m):
                    member = neighbours._get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    neighbours_getter._get(
                        member,
                        input_data,
                        neighbour_neighbours,
                        metric,
                        cluster_params
                        )

                    if not neighbour_neighbours._enough(cluster_params):
                        _consider[member] = 0
                        continue

                    if similarity_checker._check(
                            neighbours,
                            neighbour_neighbours,
                            cluster_params):
                        _consider[member] = 0
                        _labels[member] = current
                        queue._push(member)

                if queue._is_empty():
                    break

                point = queue._pop()
                neighbours_getter._get(
                    point,
                    input_data,
                    neighbour_neighbours,
                    metric,
                    cluster_params
                    )

            current += 1

    def fit(
            self,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_GETTER_EXT neighbours_getter,
            NEIGHBOURS_EXT neighbours,
            NEIGHBOUR_NEIGHBOURS_EXT neighbour_neighbours,
            METRIC_EXT metric,
            SIMILARITY_CHECKER_EXT similarity_checker,
            QUEUE_EXT queue,
            Labels labels,
            ClusterParameters cluster_params):

        self._fit(
            input_data,
            neighbours_getter,
            neighbours,
            neighbour_neighbours,
            metric,
            similarity_checker,
            queue,
            labels,
            cluster_params,
        )