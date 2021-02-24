from abc import ABC, abstractmethod
from typing import Type

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
            special_dummy: Type['Neighbours'] = None) -> Type['Neighbours']:
        """Return neighbours of point"""


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
            NEIGHBOURS special_dummy):

        cdef NEIGHBOURS neighbours = getter.get(index, metric)

        return neighbours


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @property
    @abstractmethod
    def is_sorted(self) -> bool:
       """Return True if neighbour indices are sorted"""

    @property
    @abstractmethod
    def is_selfcounting(self) -> bool:
       """Return True if points count as their own neighbour"""

    @abstractmethod
    def enough(self, cnn_cutoff: int) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def check_similarity(self, other: Type['Neighbours']) -> bool:
        """Return True if neighbours fullfil the density criterion"""


cdef class NeighboursExtMemoryview:
    """Implements the neighbours interface"""

    def __cinit__(self, neighbours):
        self.neighbours = neighbours
        self.n_points = self.neighbours.shape[0]

    cdef bint enough(self, AINDEX cnn_cutoff):
        if self.n_points > cnn_cutoff:
            return True
        return False

    cdef inline AINDEX get_member(self, AINDEX index) nogil:
        return self.neighbours[index]

    cdef bint check_similarity(self, NeighboursExtMemoryview other):
        return True
