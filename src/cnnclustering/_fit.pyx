from abc import ABC, abstractmethod
from collections import deque

from cython.operator cimport dereference as deref

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


class Fitter(ABC):
    """Defines the fitter interface"""

    @abstractmethod
    def fit(
        self,
        input_data,
        neighbours_getter,
        special_dummy,
        metric,
        similarity_checker,
        labels,
        consider,
        cluster_params):
        """Generic clustering"""


cdef void fit_id(
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        AINDEX* labels,
        ABOOL* consider,
        ClusterParameters* cluster_params):
    pass


cdef class FitterDeque:
    """Concrete implementation of the fitter interface"""

    cdef void fit(
            self,
            INPUT_DATA input_data,
            NEIGHBOURS_GETTER neighbours_getter,
            NEIGHBOURS special_dummy,
            METRIC metric,
            SIMILARITY_CHECKER similarity_checker,
            AINDEX* labels,
            ABOOL* consider,
            ClusterParameters* cluster_params):
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

        n = input_data.n_points

        current = 1
        q = deque()  # V1 (Queue)

        for init_point in range(n):
            if consider[init_point] == 0:
                continue
            consider[init_point] = 0

            if INPUT_DATA is object:
                neighbours = input_data.get_neighbours(
                    init_point, neighbours_getter, metric,
                    deref(cluster_params), special_dummy,
                    )
            else:
                neighbours = input_data.get_neighbours(
                    init_point, neighbours_getter, metric,
                    cluster_params, special_dummy,
                    )

            if NEIGHBOURS is object:
                if not neighbours.enough(
                        deref(cluster_params)):
                    consider[member] = 0
                    continue
            else:
                if not neighbours.enough(cluster_params):
                    continue

            labels[init_point] = current

            while True:

                m = neighbours.n_points

                for member_index in range(m):
                    member = neighbours.get_member(member_index)

                    if consider[member] == 0:
                        continue

                    if INPUT_DATA is object:
                        neighbour_neighbours = input_data.get_neighbours(
                            member, neighbours_getter, metric,
                            deref(cluster_params), special_dummy
                            )
                    else:
                        neighbour_neighbours = input_data.get_neighbours(
                            member, neighbours_getter, metric,
                            cluster_params, special_dummy
                            )

                    if NEIGHBOURS is object:
                        if not neighbour_neighbours.enough(
                                deref(cluster_params)):
                            consider[member] = 0
                            continue

                        if neighbours.check_similarity(
                                neighbour_neighbours,
                                similarity_checker,
                                deref(cluster_params)):
                            consider[member] = 0
                            labels[member] = current
                            q.append(member)
                    else:
                        if not neighbour_neighbours.enough(cluster_params):
                            consider[member] = 0
                            continue

                        if neighbours.check_similarity(
                                neighbour_neighbours,
                                similarity_checker,
                                cluster_params):
                            consider[member] = 0
                            labels[member] = current
                            q.append(member)

                if not q:
                    break

                point = q.popleft()
                if INPUT_DATA is object:
                    neighbours = input_data.get_neighbours(
                        point, neighbours_getter, metric,
                        deref(cluster_params), special_dummy
                        )
                else:
                    neighbours = input_data.get_neighbours(
                        point, neighbours_getter, metric,
                        cluster_params, special_dummy
                        )

            current += 1
