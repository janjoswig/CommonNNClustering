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
    SimilarityChecker
)

class Fitter(ABC):
    """Defines the fitter interface"""

    @abstractmethod
    def fit(
        self,
        input_data: Type['InputData'],
        neighbours_getter: Type['NeighboursGetter'],
        special_dummy: Type['Neighbours'],
        metric: Type['Metric'],
        similarity_checker: Type['SimilarityChecker'],
        labels: Type['Labels'],
        cluster_params: Type['ClusterParameters']):
        """Generic clustering"""


cdef void fit_id(
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        Labels labels,
        ClusterParameters cluster_params):
    pass


cdef class FitterExtDeque:
    """Concrete implementation of the fitter interface"""

    cdef void fit(
            self,
            INPUT_DATA input_data,
            NEIGHBOURS_GETTER neighbours_getter,
            NEIGHBOURS special_dummy,
            METRIC metric,
            SIMILARITY_CHECKER similarity_checker,
            Labels labels,
            ClusterParameters cluster_params):
        """Generic clustering

        Features of this variant:
            V1 (Queue): Python :obj:`collections.deque`

        Args:
            input_data: Data source implementing the input data interface.
            labels: Array as cluster label assignment container.
            consider: Boolean array to track point inclusion.
        """

        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef object q
        cdef NEIGHBOURS neighbours, neighbour_neighbours
        cdef AINDEX* _labels = &labels.labels[0]
        cdef ABOOL* _consider = &labels.consider[0]

        n = input_data.n_points

        current = 1
        q = deque()  # V1 (Queue)

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            neighbours = neighbours_getter.get(
                init_point, input_data, metric,
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

                    neighbour_neighbours = neighbours_getter.get(
                        member, input_data, metric,
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
                        q.append(member)

                if not q:
                    break

                point = q.popleft()
                neighbours = neighbours_getter.get(
                    point, input_data, metric,
                    cluster_params
                    )

            current += 1
