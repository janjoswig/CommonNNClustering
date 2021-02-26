cimport numpy as np

from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL


ctypedef fused INPUT_DATA:
    InputDataExtPointsMemoryview
    object

ctypedef fused INPUT_DATA_EXT:
    InputDataExtPointsMemoryview

ctypedef fused NEIGHBOURS:
    NeighboursExtMemoryview
    object

ctypedef fused NEIGHBOURS_EXT:
    NeighboursExtMemoryview

ctypedef fused NEIGHBOURS_GETTER:
    NeighboursGetterFromPointsMemoryview
    object

ctypedef fused METRIC:
    object

ctypedef fused SIMILARITY_CHECKER:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains
    object

ctypedef fused SIMILARITY_CHECKER_EXT:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains


cdef class ClusterParameters:
    cdef public:
        AVALUE radius_cutoff
        AINDEX cnn_cutoff


cdef class Labels:
    cdef AINDEX[::1] labels
    cdef ABOOL[::1] consider


cdef class InputDataExtPointsMemoryview:
    cdef public:
        AVALUE[:, ::1] data
        AINDEX n_points


cdef class NeighboursExtMemoryview:

    cdef public:
        AINDEX[::1] neighbours
        AINDEX n_points

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

    cdef inline bint check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class SimilarityCheckerExtSwitchContains:
    """Implements the similarity checker interface"""

    cdef inline bint check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil
