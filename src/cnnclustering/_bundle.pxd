from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL
from cnnclustering._types cimport Labels, ReferenceIndices

cdef class Bundle:
    cdef public:
        object _input_data
        object _graph
        object _labels
        object _reference_indices
        dict _children
        str alias
        object _parent
        dict meta
        object _summary
        AINDEX _hierarchy_level

    cdef object __weakref__
    cpdef void isolate(
        self,
        bint purge=*,
        bint isolate_input_data=*)
    cpdef void add_child(self, AINDEX label)

cpdef void isolate(
    Bundle bundle,
    bint purge=*,
    bint isolate_input_data=*)