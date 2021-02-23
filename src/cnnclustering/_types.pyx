from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE


class InputData(ABC):

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def get_neighbours(self, index: int) -> Type['Neighbours']:
        """Return neighbours of point"""


cdef class InputDataExtPointsMemoryview:

    def __cinit__(self, points):
        self.points = points
        self.n_points = self.points.shape[0]

    cdef NeighboursExtMemoryview get_neighbours(self, AINDEX index):
        return NeighboursExtMemoryview(np.array([0]))


class Neighbours(ABC):

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def enough(self) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_indexable(self):
       """Return indexable neighbours container"""

    @abstractmethod
    def check_similarity(self, other: Type['Neighbours']) -> bool:
        """Return True if neighbours fullfil the density criterion"""


cdef class NeighboursExtMemoryview:

    def __cinit__(self, neighbours):
        self.neighbours
        self.n_points = self.neighbours.shape[0]

    cdef bint enough(self):
        return True

    cdef AINDEX* get_indexable(self):
        return &self.neighbours[0]

    cdef bint check_similarity(self, NeighboursExtMemoryview other):
        return True
