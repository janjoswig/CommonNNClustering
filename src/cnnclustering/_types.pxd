cimport numpy as np

from cnnclustering._primitive_types cimport AINDEX, AVALUE


ctypedef fused INPUT_DATA:
    InputDataExtPointsMemoryview
    object


cdef class InputDataExtPointsMemoryview:
    cdef AVALUE[:, ::1] points
    cdef AINDEX n_points

    cdef NeighboursExtMemoryview get_neighbours(self, AINDEX index)


cdef class NeighboursExtMemoryview:

    cdef AINDEX[::1] neighbours
    cdef AINDEX n_points

    cdef bint enough(self)
    cdef AINDEX* get_indexable(self)
    cdef bint check_similarity(self, NeighboursExtMemoryview other)