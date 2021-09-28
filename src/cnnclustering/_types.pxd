cimport numpy as np

from libcpp.vector cimport vector as cppvector
from libcpp.set cimport set as cppset
from libcpp.queue cimport queue as cppqueue
from libcpp.unordered_set cimport unordered_set as cppunordered_set

from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL

cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value) nogil


ctypedef fused INPUT_DATA:
    InputDataExtComponentsMemoryview
    InputDataExtNeighbourhoodsMemoryview
    object

ctypedef fused INPUT_DATA_EXT:
    InputDataExtComponentsMemoryview
    InputDataExtNeighbourhoodsMemoryview

ctypedef fused INPUT_DATA_COMPONENTS:
    InputDataExtComponentsMemoryview
    object

ctypedef fused INPUT_DATA_EXT_COMPONENTS:
    InputDataExtComponentsMemoryview

ctypedef fused INPUT_DATA_NEIGHBOURHOODS:
    InputDataExtNeighbourhoodsMemoryview
    object

ctypedef fused INPUT_DATA_EXT_NEIGHBOURHOODS:
    InputDataExtNeighbourhoodsMemoryview

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

ctypedef fused DISTANCE_GETTER:
    DistanceGetterExtMetric
    DistanceGetterExtLookup
    object

ctypedef fused DISTANCE_GETTER_EXT:
    DistanceGetterExtMetric
    DistanceGetterExtLookup

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
        AINDEX similarity_cutoff
        AVALUE similarity_cutoff_continuous
        AINDEX n_member_cutoff
        AINDEX current_start


cdef class Labels:
    cdef public:
        dict meta

    cdef:
        AINDEX[::1] _labels
        ABOOL[::1] _consider

        cppunordered_set[AINDEX] _consider_set


cdef class InputDataExtInterface:
    cdef public:
        AINDEX n_points
        AINDEX n_dim
        dict meta

    cdef AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil
    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil
    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil
    cdef AVALUE _get_distance(self, const AINDEX point_a, const AINDEX point_b) nogil
    cdef void _compute_distances(self, InputDataExtInterface input_data) nogil
    cdef void _compute_neighbourhoods(
            self,
            InputDataExtInterface input_data, AVALUE r,
            ABOOL is_sorted, ABOOL is_selfcounting) nogil


cdef class NeighboursExtInterface:
    cdef public:
        AINDEX n_points

    cdef void _assign(self, const AINDEX member) nogil
    cdef void _reset(self) nogil
    cdef bint _enough(self, const AINDEX member_cutoff) nogil
    cdef AINDEX _get_member(self, const AINDEX index) nogil
    cdef bint _contains(self, const AINDEX member) nogil


cdef class NeighboursGetterExtInterface:
    cdef public:
        bint is_selfcounting
        bint is_sorted

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil


cdef class DistanceGetterExtInterface:
    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil


cdef class MetricExtInterface:
    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class SimilarityCheckerExtInterface:

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class QueueExtInterface:

    cdef void _push(self, const AINDEX value) nogil
    cdef AINDEX _pop(self) nogil
    cdef bint _is_empty(self) nogil


cdef class InputDataExtComponentsMemoryview(InputDataExtInterface):

    cdef:
        AVALUE[:, ::1] _data

    cdef AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil


cdef class InputDataExtDistancesLinearMemoryview(InputDataExtInterface):

    cdef:
        AVALUE[::1] _data

    cdef AVALUE _get_distance(self, const AINDEX point_a, const AINDEX point_b) nogil
    cdef void _compute_distances(self, InputDataExtInterface input_data) nogil



cdef class InputDataExtNeighbourhoodsMemoryview(InputDataExtInterface):

    cdef:
        AINDEX[:, ::1] _data
        AINDEX[::1] _n_neighbours

    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil
    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil


cdef class InputDataExtNeighbourhoodsVector(InputDataExtInterface):

    cdef:
        cppvector[cppvector[AINDEX]] _data
        cppvector[AINDEX] _n_neighbours

    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil
    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil


cdef class NeighboursExtVector(NeighboursExtInterface):

    cdef:
        AINDEX _initial_size
        cppvector[AINDEX] _neighbours

    cdef void _assign(self, const AINDEX member) nogil
    cdef void _reset(self) nogil
    cdef bint _enough(self, const AINDEX member_cutoff) nogil
    cdef AINDEX _get_member(self, AINDEX index) nogil
    cdef bint _contains(self, const AINDEX member) nogil


cdef class NeighboursExtCPPSet(NeighboursExtInterface):

    cdef:
        AINDEX _initial_size
        cppset[AINDEX] _neighbours

    cdef void _assign(self, const AINDEX member) nogil
    cdef void _reset(self) nogil
    cdef bint _enough(self, const AINDEX member_cutoff) nogil
    cdef AINDEX _get_member(self, const AINDEX index) nogil
    cdef bint _contains(self, const AINDEX member) nogil


cdef class NeighboursExtCPPUnorderedSet(NeighboursExtInterface):

    cdef:
        AINDEX _initial_size
        cppunordered_set[AINDEX] _neighbours

    cdef void _assign(self, const AINDEX member) nogil
    cdef void _reset(self) nogil
    cdef bint _enough(self, const AINDEX member_cutoff) nogil
    cdef AINDEX _get_member(self, const AINDEX index) nogil
    cdef bint _contains(self, const AINDEX member) nogil


cdef class NeighboursExtVectorCPPUnorderedSet(NeighboursExtInterface):

    cdef:
        AINDEX _initial_size
        cppvector[AINDEX] _neighbours
        cppunordered_set[AINDEX] _neighbours_view

    cdef void _assign(self, const AINDEX member) nogil
    cdef void _reset(self) nogil
    cdef bint _enough(self, const AINDEX member_cutoff) nogil
    cdef AINDEX _get_member(self, const AINDEX index) nogil
    cdef bint _contains(self, const AINDEX member) nogil


cdef class NeighboursGetterExtBruteForce(NeighboursGetterExtInterface):

    cdef public:
        DistanceGetterExtInterface _distance_getter

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil


cdef class NeighboursGetterExtLookup(NeighboursGetterExtInterface):

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil


cdef class DistanceGetterExtMetric(DistanceGetterExtInterface):
    cdef public:
        MetricExtInterface _metric

    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil


cdef class DistanceGetterExtLookup(DistanceGetterExtInterface):

    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil


cdef class MetricExtDummy(MetricExtInterface):

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtPrecomputed(MetricExtInterface):
    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtEuclidean(MetricExtInterface):
    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtEuclideanReduced(MetricExtInterface):
    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class MetricExtEuclideanPeriodicReduced(MetricExtInterface):
    cdef:
        AVALUE[::1] _bounds

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class SimilarityCheckerExtContains(SimilarityCheckerExtInterface):
    """Implements the similarity checker interface"""

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class SimilarityCheckerExtSwitchContains(SimilarityCheckerExtInterface):
    """Implements the similarity checker interface"""

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class SimilarityCheckerExtScreensorted(SimilarityCheckerExtInterface):
    """Implements the similarity checker interface"""

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil


cdef class QueueExtLIFOVector(QueueExtInterface):
    """Implements the queue interface"""

    cdef:
        cppvector[AINDEX] _queue

    cdef void _push(self, const AINDEX value) nogil
    cdef AINDEX _pop(self) nogil
    cdef bint _is_empty(self) nogil


cdef class QueueExtFIFOQueue(QueueExtInterface):
    """Implements the queue interface"""

    cdef:
        cppqueue[AINDEX] _queue

    cdef void _push(self, const AINDEX value) nogil
    cdef AINDEX _pop(self) nogil
    cdef bint _is_empty(self) nogil
