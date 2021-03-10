cimport numpy as np

from libcpp.vector cimport vector as cppvector
from libcpp.queue cimport queue as cppqueue

from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL


ctypedef fused INPUT_DATA:
    InputDataExtPointsMemoryview
    object

ctypedef fused INPUT_DATA_EXT:
    InputDataExtPointsMemoryview

ctypedef fused NEIGHBOURS:
    NeighboursExtVector
    object

ctypedef fused NEIGHBOURS_EXT:
    NeighboursExtVector

ctypedef fused NEIGHBOUR_NEIGHBOURS:
    NeighboursExtVector
    object

ctypedef fused NEIGHBOUR_NEIGHBOURS_EXT:
    NeighboursExtVector

ctypedef fused NEIGHBOURS_GETTER:
    NeighboursGetterExtBruteForce
    NeighboursGetterExtLookup
    object

ctypedef fused NEIGHBOURS_GETTER_EXT:
    NeighboursGetterExtBruteForce
    # NeighboursGetterExtLookup

ctypedef fused METRIC:
    MetricExtDummy
    MetricExtPrecomputed
    MetricExtEuclidean
    MetricExtEuclideanReduced
    object

ctypedef fused METRIC_EXT:
    MetricExtDummy
    MetricExtPrecomputed
    MetricExtEuclidean
    MetricExtEuclideanReduced


ctypedef fused SIMILARITY_CHECKER:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains
    object

ctypedef fused SIMILARITY_CHECKER_EXT:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains

ctypedef fused QUEUE:
    QueueExtLIFOVector
    QueueExtFIFOQueue
    object

ctypedef fused QUEUE_EXT:
    QueueExtLIFOVector
    QueueExtFIFOQueue


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

    cdef inline AVALUE _get_component(
            self, AINDEX point, AINDEX dimension) nogil
    cdef inline AINDEX _get_n_neighbours(self, AINDEX point) nogil
    cdef inline AINDEX _get_neighbour(self, AINDEX point, AINDEX member) nogil


cdef class NeighboursExtVector:

    cdef public:
        AINDEX n_points

    cdef:
        AINDEX _initial_size
        cppvector[AINDEX] _neighbours

    cdef inline void _assign(self, AINDEX member) nogil
    cdef inline void _reset(self) nogil
    cdef inline bint _enough(self, ClusterParameters cluster_params) nogil
    cdef inline AINDEX _get_member(self, AINDEX index) nogil
    cdef inline bint _contains(self, AINDEX member) nogil


cdef class NeighboursGetterExtBruteForce:

    cdef inline void _get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil


cdef class NeighboursGetterExtLookup:
    pass


cdef class MetricExtDummy:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil


cdef class MetricExtPrecomputed:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil


cdef class MetricExtEuclidean:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil


cdef class MetricExtEuclideanReduced:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil


cdef class SimilarityCheckerExtContains:
    """Implements the similarity checker interface"""

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class SimilarityCheckerExtSwitchContains:
    """Implements the similarity checker interface"""

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class QueueExtLIFOVector:
    """Implements the queue interface"""

    cdef:
        cppvector[AINDEX] _queue

    cdef inline void _push(self, AINDEX value) nogil
    cdef inline AINDEX _pop(self) nogil
    cdef inline bint _is_empty(self) nogil


cdef class QueueExtFIFOQueue:
    """Implements the queue interface"""

    cdef:
        cppqueue[AINDEX] _queue

    cdef inline void _push(self, AINDEX value) nogil
    cdef inline AINDEX _pop(self) nogil
    cdef inline bint _is_empty(self) nogil
