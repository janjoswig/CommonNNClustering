from cnnclustering._primitive_types cimport AVALUE, AINDEX, ABOOL
from cnnclustering._types cimport ClusterParameters, Labels
from cnnclustering._types cimport (
    INPUT_DATA,
    NEIGHBOURS_GETTER,
    NEIGHBOURS, 
    METRIC,
    SIMILARITY_CHECKER,
    QUEUE
    )
from cnnclustering._types cimport (
    INPUT_DATA_EXT,
    NEIGHBOURS_GETTER_EXT,
    NEIGHBOURS_EXT, 
    METRIC_EXT,
    SIMILARITY_CHECKER_EXT,
    QUEUE_EXT
    )


ctypedef fused FITTER:
    FitterExtBFS
    object

ctypedef fused FITTER_EXT:
    FitterExtBFS


cdef class FitterExtBFS:
    cdef void fit(
        self,
        INPUT_DATA_EXT input_data,
        NEIGHBOURS_GETTER_EXT neighbours_getter,
        NEIGHBOURS_EXT special_dummy,
        METRIC_EXT metric,
        SIMILARITY_CHECKER_EXT similarity_checker,
        QUEUE_EXT queue,
        Labels labels,
        ClusterParameters cluster_params)
