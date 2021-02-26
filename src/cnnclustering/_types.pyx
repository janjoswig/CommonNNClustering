from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Type

import numpy as np

from cython.operator cimport dereference as deref

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class ClusterParameters:
    def __cinit__(self, radius_cutoff, cnn_cutoff):
        self.radius_cutoff = radius_cutoff
        self.cnn_cutoff = cnn_cutoff


cdef class Labels:
    def __cinit__(self, labels, consider=None):
        self.labels = labels
        if consider is None:
            self.consider = np.ones_like(self.labels, dtype=P_ABOOL)
        else:
            self.consider = consider

            if self.labels.shape[0] != self.consider.shape[0]:
                raise ValueError(
                    "'labels' and 'consider' must have the same length"
                    )


class InputData(ABC):
    """Defines the input data interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""


class InputDataNeighboursSequence(InputData):
    """Implements the input data interface

    Neighbours of points stored as a sequence.
    """

    def __init__(self, data: Sequence):
        self._data = data

    @property
    def n_points(self):
        return len(self._data)


cdef class InputDataExtPointsMemoryview:
    """Implements the input data interface"""

    def __cinit__(self, data):
        self.data = data
        self.n_points = self.points.shape[0]


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def enough(self, cluster_params: Type['ClusterParameters']) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def contains(self, member: int) -> bool:
       """Return True if member is in neighbours container"""


class NeighboursSequence(Neighbours):
    def __init__(
            self,
            neighbours: Sequence):
        self._neighbours = neighbours

    @property
    def n_points(self):
        return len(self._neighbours)

    def enough(self, cluster_params: Type['ClusterParameters']):
        if self.n_points > cluster_params.cnn_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        return self._neighbours[index]

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


cdef class NeighboursExtMemoryview:
    """Implements the neighbours interface"""

    def __cinit__(self, neighbours):
        self.neighbours = neighbours
        self.n_points = self.neighbours.shape[0]

    cdef bint enough(self, ClusterParameters cluster_params):
        if self.n_points > cluster_params.cnn_cutoff:
            return True
        return False

    cdef inline AINDEX get_member(self, AINDEX index) nogil:
        return self.neighbours[index]

    cdef inline bint contains(self, AINDEX member) nogil:
        cdef AINDEX index

        for index in range(self.n_points):
            if self.neighbours[index] == member:
                return True
        return False


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


    @property
    @abstractmethod
    def neighbours_dummy(self) -> Type['Neighbours']:
       """Return dummy instance of neighbours object this getter will create"""

    @abstractmethod
    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):
        """Return neighbours for point in input data"""


class NeighboursGetterFromNeighboursSequenceToSequence(NeighboursGetter):

    def __init__(self, is_sorted=False, is_selfcounting=False):
        self.is_sorted = is_sorted
        self.is_selfcounting = is_selfcounting
        self._neighbours_dummy = NeighboursSequence

    @property
    def neighbours_dummy(self):
        return self._neighbours_dummy([])

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):
        return self._neighbours_dummy(input_data[index])


cdef class NeighboursGetterFromPointsMemoryview:
    def __init__(self):
        # sorted?
        # self_counting?
        self.neighbours_dummy = NeighboursExtMemoryview(
            np.array([], dtype=P_AINDEX)
            )

    cdef NeighboursExtMemoryview get(
            self,
            AINDEX index,
            INPUT_DATA input_data,
            METRIC metric,
            ClusterParameters cluster_params):
        return NeighboursExtMemoryview(np.array([], dtype=P_AINDEX))


class Metric(ABC):
    """Defines the metric-interface"""
    pass


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

    cdef bint check(
            self,
            NEIGHBOURS neighbours_a,
            NEIGHBOURS neighbours_b,
            ClusterParameters cluster_params):

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

    cdef bint check(
            self,
            NEIGHBOURS neighbours_a,
            NEIGHBOURS neighbours_b,
            ClusterParameters cluster_params):

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        if nb < na:
           neighbours_a, neighbours_b = neighbours_b, neighbours_a

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                break
        return False
