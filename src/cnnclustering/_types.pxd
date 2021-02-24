cimport numpy as np

from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL


cdef struct ClusterParameters:
    AVALUE radius_cutoff
    AINDEX cnn_cutoff

ctypedef fused INPUT_DATA:
    InputDataExtPointsMemoryview
    object

ctypedef fused NEIGHBOURS:
    NeighboursExtMemoryview
    object

ctypedef fused NEIGHBOURS_GETTER:
    object

ctypedef fused METRIC:
    object


cdef class InputDataExtPointsMemoryview:
    cdef AVALUE[:, ::1] points
    cdef AINDEX n_points

    cdef NEIGHBOURS get_neighbours(
        self,
        AINDEX index,
        NEIGHBOURS_GETTER getter,
        METRIC metric,
        NEIGHBOURS special_dummy)


cdef class NeighboursExtMemoryview:

    cdef AINDEX[::1] neighbours
    cdef AINDEX n_points
    cdef bint is_sorted
    cdef bint is_selfcounting

    cdef bint enough(self, AINDEX cnn_cutoff)
    cdef inline AINDEX get_member(self, AINDEX index) nogil
    cdef bint check_similarity(self, NeighboursExtMemoryview other)
