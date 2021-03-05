from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Any, Type

import numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport sqrt as csqrt, pow as cpow

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class ClusterParameters:
    def __cinit__(self, radius_cutoff, cnn_cutoff):
        self.radius_cutoff = radius_cutoff
        self.cnn_cutoff = cnn_cutoff

    def to_dict(self):
        return {
            "radius_cutoff": self.radius_cutoff,
            "cnn_cutoff": self.cnn_cutoff
            }

    def __repr__(self):
        return f"{self.to_dict()!r}"

    def __str__(self):
        return f"{self.to_dict()!s}"


cdef class Labels:
    def __cinit__(self, labels, consider=None):
        self._labels = labels
        if consider is None:
            self._consider = np.ones_like(self._labels, dtype=P_ABOOL)
        else:
            self._consider = consider

            if self._labels.shape[0] != self._consider.shape[0]:
                raise ValueError(
                    "'labels' and 'consider' must have the same length"
                    )

    @property
    def labels(self):
        return np.asarray(self._labels)

    @property
    def consider(self):
        return np.asarray(self._consider)

    def __repr__(self):
        return f"{type(self).__name__}({list(self.labels)!s})"

    def __str__(self):
        return f"{self.labels!s}"


class InputData(ABC):
    """Defines the input data interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @property
    @abstractmethod
    def n_dim(self) -> int:
       """Return total number of dimensions"""

    @abstractmethod
    def get_component(self, point: int, dimension: int) -> float:
       """Return one component of point coordinates"""


class InputDataNeighboursSequence(InputData):
    """Implements the input data interface

    Neighbours of points stored as a sequence.
    """

    def __init__(self, data: Sequence):
        self._data = data

    @property
    def n_points(self):
        return len(self._data)

    @property
    def n_dim(self):
        return None

    def get_component(self, point: int, dimension: int) -> float:
        return None


cdef class InputDataExtPointsMemoryview:
    """Implements the input data interface"""

    def __cinit__(self, AVALUE[:, ::1] data not None):
        self._data = data
        self.n_points = self._data.shape[0]
        self.n_dim = self._data.shape[1]

    @property
    def data(self):
       return np.asarray(self._data)

    cdef inline AVALUE _get_component(
            self, AINDEX point, AINDEX dimension) nogil:
        return self._data[point, dimension]

    def get_component(
            self, point: int, dimension: int) -> float:
        return self._get_component(point, dimension)


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def assign(self, member: int) -> None:
       """Add a member to this container"""

    @abstractmethod
    def reset(self) -> None:
       """Reset/empty this container"""

    @abstractmethod
    def enough(self, cluster_params: Type['ClusterParameters']) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def contains(self, member: int) -> bool:
       """Return True if member is in neighbours container"""


class NeighboursList(Neighbours):
    def __init__(self, neighbours=None):
        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
        else:
            self.reset()

    @property
    def n_points(self):
        return self._n_points

    def assign(self, member: int):
        self._neighbours.append(member)
        self._n_points += 1

    def reset(self):
        self._neighbours = []
        self._n_points = 0

    def enough(self, cluster_params: Type['ClusterParameters']):
        if self._n_points > cluster_params.cnn_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        return self._neighbours[index]

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


class NeighboursSet(Neighbours):
    def __init__(self, neighbours=None):
        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
        else:
            self.reset()

    @property
    def n_points(self):
        return self._n_points

    def assign(self, member: int):
        self._neighbours.add(member)
        self._n_points += 1

    def reset(self):
        self._neighbours = {}
        self._n_points = 0
        self._query = 0
        self._iter = None

    def enough(self, cluster_params: Type['ClusterParameters']):
        if self._n_points > cluster_params.cnn_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        if (self._iter is None) or (index != self._query):
            self._iter = iter(self._neighbours)
            self._query = 0

        while self._query != index:
            _ = next(self._iter)
        return next(self._iter)

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


cdef class NeighboursExtVector:
    """Implements the neighbours interface"""

    def __cinit__(self, AINDEX initial_size, neighbours=None):
        self._initial_size = initial_size

        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
            self._neighbours.reserve(self._initial_size)
        else:
            self._reset()

    cdef inline void _assign(self, AINDEX member) nogil:
        self._neighbours.push_back(member)
        self.n_points += 1

    def assign(self, member: int):
        self._assign(member)

    cdef inline void _reset(self) nogil:
        self._neighbours.resize(0)
        self._neighbours.reserve(self._initial_size)
        self.n_points = 0

    def reset(self):
        self._reset()

    cdef inline bint _enough(self, ClusterParameters cluster_params) nogil:
        if self.n_points > cluster_params.cnn_cutoff:
            return True
        return False

    def enough(self, cluster_params: Type["ClusterParameters"]):
        self._enough(cluster_params)

    cdef inline AINDEX _get_member(self, AINDEX index) nogil:
        return self._neighbours[index]

    def get_member(self, index: int):
        return self._get_member(index)

    cdef inline bint _contains(self, AINDEX member) nogil:
        cdef AINDEX index

        for index in range(self.n_points):
            if self._neighbours[index] == member:
                return True
        return False

    def contains(self, member: int):
        return self._contains(member)


class NeighboursGetter(ABC):
    """Defines the neighbours-getter interface"""

    @property
    @abstractmethod
    def is_sorted(self) -> bool:
       """Return True if neighbour indices are sorted"""

    @property
    @abstractmethod
    def is_selfcounting(self) -> bool:
       """Return True if points count as their own neighbour"""

    @abstractmethod
    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours for point in input data"""


class NeighboursGetterLookup(NeighboursGetter):

    def __init__(self, is_sorted=False, is_selfcounting=False):
        self.is_sorted = is_sorted
        self.is_selfcounting = is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:

        neighbours.reset()

        cdef AINDEX i
        for i in input_data[index]:
            neighbours.assign(i)


class NeighboursGetterBruteForce(NeighboursGetter):

    def __init__(self):
        self.is_sorted = False
        self.is_selfcounting = True

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i
        cdef AVALUE distance

        neighbours.reset()

        for i in range(input_data.n_points):
            distance = metric.calc_distance(index, i, input_data)

            if distance <= cluster_params.radius_cutoff:
                neighbours.assign(i)


cdef class NeighboursGetterExtBruteForce:

    cdef _get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params):

        cdef AINDEX i
        cdef AVALUE distance

        neighbours.reset()

        for i in range(input_data.n_points):
            distance = metric._calc_distance(index, i, input_data)

            if distance <= cluster_params.radius_cutoff:
                neighbours._assign(i)

    def get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params):

        self._get(
            index,
            input_data,
            neighbours,
            metric,
            cluster_params,
        )


cdef class NeighboursGetterExtLookup:
    pass


class Metric(ABC):
    """Defines the metric-interface"""

    @abstractmethod
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        """Return distance between two points in input data"""


class MetricPrecomputed(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        return input_data.get_component(index_a, index_b)


cdef class MetricExtPrecomputed:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        return input_data._get_component(index_a, index_b)

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)


class MetricEuclidean(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)


cdef class MetricExtEuclidean:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)


class MetricEuclideanReduced(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return total


cdef class MetricExtEuclideanReduced:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)


class SimilarityChecker(ABC):
    """Defines the similarity checker interface"""

    @abstractmethod
    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:
        """Retrun True if a and b have sufficiently many common neighbours"""


class SimilarityCheckerContains(SimilarityChecker):
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        no switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerSwitchContains`).
    """

    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:

        cdef AINDEX na = neighbours_a.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                break
        return False


class SimilarityCheckerSwitchContains(SimilarityChecker):
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerContains`).
    """

    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        if nb < na:
            neighbours_a, neighbours_b = neighbours_b, neighbours_a
            na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                break
        return False


cdef class SimilarityCheckerExtContains:
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        no switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerExtSwitchContains`).
    """

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX na = neighbours_a.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1
                if common == c:
                    return True
                break
        return False

    def check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params):

        return self._check(neighbours_a, neighbours_b, cluster_params)


cdef class SimilarityCheckerExtSwitchContains:
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerExtContains`).
    """

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        if nb < na:
            with gil:
                neighbours_a, neighbours_b = neighbours_b, neighbours_a
                na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1
                if common == c:
                    return True
                break
        return False

    def check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params):

        return self._check(neighbours_a, neighbours_b, cluster_params)


class Queue(ABC):
    """Defines the queue interface"""

    @abstractmethod
    def push(self, value):
        """Put value into the queue"""

    @abstractmethod
    def pop(self):
        """Retrieve value from the queue"""

    @abstractmethod
    def is_empty(self):
        """Return True if there are no values in the queue"""


class QueueFIFODeque(Queue):
    """Implements the queue interface"""

    def __init__(self):
       self._queue = deque()

    def push(self, value):
        """Append value to back/right end"""
        self._queue.append(value)

    def pop(self):
        """Retrieve value from front/left end"""
        return self._queue.popleft()

    def is_empty(self):
        """Return True if there are no values in the queue"""
        if self._queue:
            return False
        return True


cdef class QueueExtLIFOVector:
    """Implements the queue interface"""

    cdef inline void _push(self, AINDEX value) nogil:
        """Append value to back/right end"""
        self._queue.push_back(value)

    cdef inline AINDEX _pop(self) nogil:
        """Retrieve value from back/right end"""

        cdef AINDEX value = self._queue.back()
        self._queue.pop_back()

        return value

    cdef inline bint _is_empty(self) nogil:
        """Return True if there are no values in the queue"""
        return self._queue.empty()

    def push(self, value: int):
        self._push(value)

    def pop(self) -> int:
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()


cdef class QueueExtFIFOQueue:
    """Implements the queue interface"""

    cdef inline void _push(self, AINDEX value) nogil:
        """Append value to back/right end"""
        self._queue.push(value)

    cdef inline AINDEX _pop(self) nogil:
        """Retrieve value from back/right end"""

        cdef AINDEX value = self._queue.front()
        self._queue.pop()

        return value

    cdef inline bint _is_empty(self) nogil:
        """Return True if there are no values in the queue"""
        return self._queue.empty()

    def push(self, value: int):
        self._push(value)

    def pop(self) -> int:
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()
