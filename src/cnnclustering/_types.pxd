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
    NeighboursGetterExtBruteForce
    NeighboursGetterExtLookup
    object

ctypedef fused NEIGHBOURS_GETTER_EXT:
    NeighboursGetterExtBruteForce
    NeighboursGetterExtLookup

ctypedef fused METRIC:
    MetricExtEuclideanReduced
    MetricExtPrecomputed
    object

ctypedef fused METRIC_EXT:
    MetricExtEuclideanReduced
    MetricExtPrecomputed

ctypedef fused SIMILARITY_CHECKER:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains
    object

ctypedef fused SIMILARITY_CHECKER_EXT:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains

ctypedef fused QUEUE:
    QueueExtVector
    object

ctypedef fused QUEUE_EXT:
    QueueExtVector


cdef class ClusterParameters:
    cdef public:
        AVALUE radius_cutoff
        AINDEX cnn_cutoff


cdef class Labels:
    cdef:
        AINDEX[::1] _labels
        ABOOL[::1] _consider


cdef class InputDataExtPointsMemoryview:
    cdef public:
        AINDEX n_points
        AINDEX n_dim

    cdef:
        AVALUE[:, ::1] _data

    cdef inline AVALUE get_component(
            self, AINDEX point, AINDEX dimension) nogil


cdef class NeighboursExtMemoryview:

    cdef public:
        AINDEX n_points

    cdef:
        AINDEX[::1] neighbours

    cdef bint enough(self, ClusterParameters cluster_params)
    cdef inline AINDEX get_member(self, AINDEX index) nogil
    cdef inline bint contains(self, AINDEX member) nogil


cdef class NeighboursGetterExtBruteForce:
    cdef NeighboursExtMemoryview neighbours_dummy
 
    cdef NeighboursExtMemoryview get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            METRIC_EXT metric,
            ClusterParameters cluster_params)


cdef class NeighboursGetterExtLookup:
    pass


cdef class MetricExtPrecomputed:
    cdef inline AVALUE calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil


cdef class MetricExtEuclideanReduced:
    cdef inline AVALUE calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil


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


cdef class QueueExtVector:
    pass