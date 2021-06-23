cimport numpy as np

from libcpp.vector cimport vector as cppvector
from libcpp.set cimport set as cppset
from libcpp.queue cimport queue as cppqueue
from libcpp.unordered_set cimport unordered_set as cppunordered_set

from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL


ctypedef fused INPUT_DATA:
    InputDataExtPointsMemoryview
    InputDataExtNeighboursMemoryview
    object

ctypedef fused INPUT_DATA_EXT:
    InputDataExtPointsMemoryview
    InputDataExtNeighboursMemoryview

ctypedef fused NEIGHBOURS:
    NeighboursExtVector
    NeighboursExtVectorCPPUnorderedSet
    NeighboursExtCPPSet
    NeighboursExtCPPUnorderedSet
    object

ctypedef fused NEIGHBOURS_EXT:
    NeighboursExtVector
    NeighboursExtVectorCPPUnorderedSet
    NeighboursExtCPPSet
    NeighboursExtCPPUnorderedSet

ctypedef fused NEIGHBOUR_NEIGHBOURS:
    NeighboursExtVector
    NeighboursExtVectorCPPUnorderedSet
    NeighboursExtCPPSet
    NeighboursExtCPPUnorderedSet
    object

ctypedef fused NEIGHBOUR_NEIGHBOURS_EXT:
    NeighboursExtVector
    NeighboursExtVectorCPPUnorderedSet
    NeighboursExtCPPSet
    NeighboursExtCPPUnorderedSet

ctypedef fused NEIGHBOURS_GETTER:
    NeighboursGetterExtBruteForce
    NeighboursGetterExtLookup
    object

ctypedef fused NEIGHBOURS_GETTER_EXT:
    NeighboursGetterExtBruteForce
    NeighboursGetterExtLookup

ctypedef fused METRIC:
    MetricExtDummy
    MetricExtPrecomputed
    MetricExtEuclidean
    MetricExtEuclideanReduced
    MetricExtEuclideanPeriodicReduced
    object

ctypedef fused METRIC_EXT:
    MetricExtDummy
    MetricExtPrecomputed
    MetricExtEuclidean
    MetricExtEuclideanReduced
    MetricExtEuclideanPeriodicReduced


ctypedef fused SIMILARITY_CHECKER:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains
    SimilarityCheckerExtScreensorted
    object

ctypedef fused SIMILARITY_CHECKER_EXT:
    SimilarityCheckerExtContains
    SimilarityCheckerExtSwitchContains
    SimilarityCheckerExtScreensorted

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
        AINDEX current_start


cdef class Labels:
    cdef public:
        dict meta

    cdef:
        AINDEX[::1] _labels
        ABOOL[::1] _consider

        cppunordered_set[AINDEX] _consider_set


cdef class InputDataExtPointsMemoryview:
    cdef public:
        AINDEX n_points
        AINDEX n_dim
        dict meta

    cdef:
        AVALUE[:, ::1] _data

    cdef inline AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil
    cdef inline AINDEX _get_n_neighbours(self, const AINDEX point) nogil
    cdef inline AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil


cdef class InputDataExtNeighboursMemoryview:
    cdef public:
        AINDEX n_points
        AINDEX n_dim
        dict meta

    cdef:
        AINDEX[:, ::1] _data
        AINDEX[::1] _n_neighbours

    cdef inline AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil
    cdef inline AINDEX _get_n_neighbours(self, const AINDEX point) nogil
    cdef inline AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil


cdef class NeighboursExtVector:

    cdef public:
        AINDEX n_points

    cdef:
        AINDEX _initial_size
        cppvector[AINDEX] _neighbours

    cdef inline void _assign(self, const AINDEX member) nogil
    cdef inline void _reset(self) nogil
    cdef inline bint _enough(self, const AINDEX member_cutoff) nogil
    cdef inline AINDEX _get_member(self, AINDEX index) nogil
    cdef inline bint _contains(self, const AINDEX member) nogil


cdef class NeighboursExtCPPSet:

    cdef public:
        AINDEX n_points

    cdef:
        AINDEX _initial_size
        cppset[AINDEX] _neighbours

    cdef inline void _assign(self, const AINDEX member) nogil
    cdef inline void _reset(self) nogil
    cdef inline bint _enough(self, const AINDEX member_cutoff) nogil
    cdef inline AINDEX _get_member(self, const AINDEX index) nogil
    cdef inline bint _contains(self, const AINDEX member) nogil


cdef class NeighboursExtCPPUnorderedSet:

    cdef public:
        AINDEX n_points

    cdef:
        AINDEX _initial_size
        cppunordered_set[AINDEX] _neighbours

    cdef inline void _assign(self, const AINDEX member) nogil
    cdef inline void _reset(self) nogil
    cdef inline bint _enough(self, const AINDEX member_cutoff) nogil
    cdef inline AINDEX _get_member(self, const AINDEX index) nogil
    cdef inline bint _contains(self, const AINDEX member) nogil


cdef class NeighboursExtVectorCPPUnorderedSet:

    cdef public:
        AINDEX n_points

    cdef:
        AINDEX _initial_size
        cppvector[AINDEX] _neighbours
        cppunordered_set[AINDEX] _neighbours_view

    cdef inline void _assign(self, const AINDEX member) nogil
    cdef inline void _reset(self) nogil
    cdef inline bint _enough(self, const AINDEX member_cutoff) nogil
    cdef inline AINDEX _get_member(self, const AINDEX index) nogil
    cdef inline bint _contains(self, const AINDEX member) nogil


cdef class NeighboursGetterExtBruteForce:
    cdef public:
        bint is_selfcounting
        bint is_sorted

    cdef inline void _get(
            self,
            const AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil

    cdef inline void _get_other(
            self,
            const AINDEX index,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil


cdef class NeighboursGetterExtLookup:
    cdef public:
        bint is_selfcounting
        bint is_sorted

    cdef inline void _get(
            self,
            const AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil

    cdef inline void _get_other(
            self,
            const AINDEX index,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil


cdef class MetricExtDummy:
    cdef inline AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil

    cdef inline AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtPrecomputed:
    cdef inline AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil

    cdef inline AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtEuclidean:
    cdef inline AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil

    cdef inline AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtEuclideanReduced:
    cdef inline AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil

    cdef inline AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtEuclideanPeriodicReduced:
    cdef:
        AVALUE[::1] _bounds

    cdef inline AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil

    cdef inline AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


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


cdef class SimilarityCheckerExtScreensorted:
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

    cdef inline void _push(self, const AINDEX value) nogil
    cdef inline AINDEX _pop(self) nogil
    cdef inline bint _is_empty(self) nogil


cdef class QueueExtFIFOQueue:
    """Implements the queue interface"""

    cdef:
        cppqueue[AINDEX] _queue

    cdef inline void _push(self, const AINDEX value) nogil
    cdef inline AINDEX _pop(self) nogil
    cdef inline bint _is_empty(self) nogil
