from libcpp.unordered_set cimport unordered_set as cppunordered_set

from cnnclustering._primitive_types cimport AVALUE, AINDEX, ABOOL
from cnnclustering._types cimport ClusterParameters, Labels
from cnnclustering._types cimport (
    InputDataExtInterface,
    NeighboursGetterExtInterface,
    NeighboursExtInterface,
    SimilarityCheckerExtInterface,
    QueueExtInterface,
)


ctypedef fused FITTER:
    FitterExtBFS
    object

ctypedef fused FITTER_EXT:
    FitterExtBFS


cdef class FitterExtBFS:
    cdef:
        NeighboursGetterExtInterface _neighbours_getter
        NeighboursExtInterface _neighbours
        NeighboursExtInterface _neighbour_neighbours
        SimilarityCheckerExtInterface _similarity_checker
        QueueExtInterface _queue

    cdef void _fit(
        self,
        InputDataExtInterface input_data,
        Labels labels,
        ClusterParameters cluster_params) nogil


cdef class FitterExtBFSDebug:
    cdef:
        bint _verbose
        bint _yielding

    cdef:
        NeighboursGetterExtInterface _neighbours_getter
        NeighboursExtInterface _neighbours
        NeighboursExtInterface _neighbour_neighbours
        SimilarityCheckerExtInterface _similarity_checker
        QueueExtInterface _queue
