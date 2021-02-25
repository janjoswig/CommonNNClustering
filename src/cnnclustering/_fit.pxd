from cnnclustering._primitive_types cimport AVALUE, AINDEX, ABOOL
from cnnclustering._types cimport (
    INPUT_DATA,
    NEIGHBOURS, NEIGHBOURS_GETTER,
    METRIC,
    SIMILARITY_CHECKER
    )
from cnnclustering._types cimport ClusterParameters


ctypedef void (*FIT_FUN)(
    INPUT_DATA, NEIGHBOURS_GETTER, NEIGHBOURS,
    METRIC, SIMILARITY_CHECKER, AINDEX*, ABOOL*, ClusterParameters*)


ctypedef fused FITTER:
    FitterDeque
    object


cdef void fit_id(
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        AINDEX* labels,
        ABOOL* consider,
        ClusterParameters* cluster_params)


cdef class FitterDeque:
    cdef void fit(
        self,
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        AINDEX* labels,
        ABOOL* consider,
        ClusterParameters* cluster_params)
