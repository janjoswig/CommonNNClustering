from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Type

import numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


class InputData(ABC):
    """Defines the input data interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def get_neighbours(
            self,
            index: int,
            getter: Type['NeighboursGetter'],
            metric: Type['Metric'],
            cluster_params: dict,
            special_dummy: Type['Neighbours'] = None) -> Type['Neighbours']:
        """Return neighbours of point"""


class InputDataNeighboursSequence(InputData):
    """Implements the input data interface

    Neighbours of points stored as a sequence.
    """

    def __init__(self, data: Sequence):
        self._data = data

    @property
    def n_points(self):
        return len(self._data)

    def get_neighbours(
            self,
            index: int,
            getter: Type['NeighboursGetter'],
            metric: Type['Metric'],
            cluster_params: dict,
            special_dummy: Type['Neighbours'] = None) -> Type['Neighbours']:

        return getter.get(self._data, index, metric, cluster_params)


cdef class InputDataExtPointsMemoryview:
    """Implements the input data interface"""

    def __cinit__(self, points):
        self.points = points
        self.n_points = self.points.shape[0]

    cdef NEIGHBOURS get_neighbours(
            self,
            AINDEX index,
            NEIGHBOURS_GETTER getter,
            METRIC metric,
            ClusterParameters* cluster_params,
            NEIGHBOURS special_dummy):

        cdef NEIGHBOURS neighbours = getter.get(index, metric)

        return neighbours


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def enough(self, cluster_params: dict) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def check_similarity(
            self,
            other: Type['Neighbours'],
            checker: Type['SimilarityChecker'],
            cluster_params: dict) -> bool:
        """Return True if neighbours fullfil the density criterion"""


class NeighboursSequence(Neighbours):
    def __init__(
            self,
            neighbours: Sequence):
        self._neighbours = neighbours

    @property
    def n_points(self):
        return len(self._neighbours)

    def enough(self, cluster_params: dict):
        if self.n_points > cluster_params["cnn_cutoff"]:
            return True
        return False

    def get_member(self, index: int) -> int:
        return self._neighbours[index]

    def check_similarity(
            self,
            other: Type['Neighbours'],
            checker: Type['SimilarityChecker'],
            cluster_params: dict) -> bool:

        return checker.check(
            self._neighbours, other._neighbours, cluster_params
            )

cdef class NeighboursExtMemoryview:
    """Implements the neighbours interface"""

    def __cinit__(self, neighbours):
        self.neighbours = neighbours
        self.n_points = self.neighbours.shape[0]

    cdef bint enough(self, ClusterParameters* cluster_params):
        if self.n_points > cluster_params.cnn_cutoff:
            return True
        return False

    cdef inline AINDEX get_member(self, AINDEX index) nogil:
        return self.neighbours[index]

    cdef bint check_similarity(self,
            NeighboursExtMemoryview other,
            SIMILARITY_CHECKER checker,
            ClusterParameters* cluster_params):

        return checker.check(
            &self.neighbours[0], #!!!!!!!!!!!!!!!!!!
            &other.neighbours[0],
            cluster_params
            )


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
    def get(self, source, index, metric, cluster_params):
        """Return neighbours for point in source"""


class NeighboursGetterFromSequenceToSequence(NeighboursGetter):

    def __init__(self, is_sorted=False, is_selfcounting=False):
        self.is_sorted = is_sorted
        self.is_selfcounting = is_selfcounting
        self._neighbours_dummy = NeighboursSequence

    @property
    def neighbours_dummy(self):
        return self._neighbours_dummy([])

    def get(self, source, index, metric, cluster_params):
        return self._neighbours_dummy(source[index])


class Metric(ABC):
    """Defines the metric-interface"""
    pass


class SimilarityChecker(ABC):
    """Defines the similarity checker interface"""

    @abstractmethod
    def check(self, a: Any, b: Any, cluster_params: dict) -> bool:
        """Retrun True if a and b have """


cdef class SimilarityCheckerExtArray:
    """Implements the similarity checker interface"""
