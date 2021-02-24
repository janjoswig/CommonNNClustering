from cnnclustering._primitive_types cimport AVALUE, AINDEX, ABOOL
from cnnclustering._types cimport INPUT_DATA
from cnnclustering._types cimport NEIGHBOURS, NEIGHBOURS_GETTER, METRIC

from cnnclustering._types cimport ClusterParameters


ctypedef void (*FIT_FUN)(
    INPUT_DATA, NEIGHBOURS_GETTER, NEIGHBOURS,
    METRIC, AINDEX*, ABOOL*, ClusterParameters*)


cdef void fit_id(
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
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
        AINDEX* labels,
        ABOOL* consider,
        ClusterParameters* cluster_params)
