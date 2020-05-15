cimport cython
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np

# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

def cfit_from_neighbours(
        int cnn_cutoff,
        np.ndarray[object, ndim=1] neighbours):
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Cythonised.

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: NumPy array of length #points containing arrays of
            neighbouring point indices

    Returns:
        Labels
    """

    cdef np.npy_intp point, current, member, len_
    cdef np.ndarray[np.npy_intp, ndim=1] neigh
    cdef vector[np.npy_intp] labels, stack  # LIFO

    len_ = neighbours.shape[0]

    # Initialise labels
    labels = np.zeros(len_)

    # Track assigment
    consider = np.ones(len_, dtype=bool)

    # Start with first cluster (0 = noise)
    current = 1

    for point in range(len_):
        if not consider[point]:
            continue

        labels[point] = current  # Assign cluster label
        consider[point] = False  # Mark point as included

        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            # loop over members not included
            for member in neigh[consider[neigh]]:
                # conditional growth
                if len(np.intersect1d(
                        neigh, neighbours[member], assume_unique=True
                        )) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    stack.push_back(member)

            if stack.size() == 0:
                break
            point = stack.back()  # LIFO
            stack.pop_back()

        current += 1

    return labels
