cimport numpy as np

from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL


ctypedef fused INPUT_DATA:
    InputDataExtPointsMemoryview
    object

ctypedef fused NEIGHBOURS:
    NeighboursExtMemoryview
    object

ctypedef fused NEIGHBOURS_GETTER:
    NeighboursGetterFromPointsMemoryview
    object

ctypedef fused METRIC:
    object

ctypedef fused SIMILARITY_CHECKER:
    object


cdef class ClusterParameters:
    cdef public:
        AVALUE radius_cutoff
        AINDEX cnn_cutoff


cdef class Labels:
    cdef AINDEX[::1] labels
    cdef ABOOL[::1] consider


cdef class InputDataExtPointsMemoryview:
    cdef AVALUE[:, ::1] data
    cdef AINDEX n_points


cdef class NeighboursExtMemoryview:

    cdef AINDEX[::1] neighbours
    cdef AINDEX n_points
    cdef bint is_sorted
    cdef bint is_selfcounting

    cdef bint enough(self, ClusterParameters cluster_params)
    cdef inline AINDEX get_member(self, AINDEX index) nogil
    cdef inline bint contains(self, AINDEX member) nogil


cdef class NeighboursGetterFromPointsMemoryview:
    cdef NeighboursExtMemoryview neighbours_dummy
 
    cdef NeighboursExtMemoryview get(
            self,
            AINDEX index,
            INPUT_DATA input_data,
            METRIC metric,
            ClusterParameters cluster_params)


cdef class SimilarityCheckerExtContains:
    """Implements the similarity checker interface"""

    cdef bint check(
            self,
            NEIGHBOURS neighbours_a,
            NEIGHBOURS neighbours_b,
            ClusterParameters cluster_params)


cdef class SimilarityCheckerExtSwitchContains:
    """Implements the similarity checker interface"""

    cdef bint check(
            self,
            NEIGHBOURS neighbours_a,
            NEIGHBOURS neighbours_b,
            ClusterParameters cluster_params)
