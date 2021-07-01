from cnnclustering._primitive_types cimport AINDEX, AVALUE, ABOOL


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
