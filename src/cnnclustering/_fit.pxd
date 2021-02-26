from cnnclustering._primitive_types cimport AVALUE, AINDEX, ABOOL
from cnnclustering._types cimport (
    INPUT_DATA,
    NEIGHBOURS_GETTER,
    NEIGHBOURS, 
    METRIC,
    SIMILARITY_CHECKER
    )
from cnnclustering._types cimport ClusterParameters, Labels


ctypedef void (*FIT_FUN)(
    INPUT_DATA, NEIGHBOURS_GETTER, NEIGHBOURS,
    METRIC, SIMILARITY_CHECKER, Labels, ClusterParameters)


ctypedef fused FITTER:
    FitterExtDeque
    object

cdef void fit_id(
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        Labels labels,
        ClusterParameters cluster_params)


cdef class FitterExtDeque:
    cdef void fit(
        self,
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        Labels labels,
        ClusterParameters cluster_params)
