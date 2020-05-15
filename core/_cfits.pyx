# cython: boundscheck = False
# cython: wraparound = False

cimport cython
from libcpp.vector cimport vector
cimport numpy as np


cdef inline bint check_similarity_set(set a, set b, int c):
    cdef int x
    cdef int common

    if c == 0:
        return 1

    common = 0
    for x in a:
        if x in b:
            common += 1
            if common == c:
                return 1
    return 0


def fit_from_neighbours(
        cnn_cutoff: int,
        neighbourhoods: list):
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbourhoods.
    Uses Python standard library only.

    Args:
        cnn_cutoff: Points need to share at least this many neighbours
            to be assigned to the same cluster (similarity criterion).
        neighbourhoods: List of length #points containing sets of
            neighbouring point indices

    Returns:
        Labels
    """

    cdef int n, m, k
    cdef vector[int] labels
    cdef vector[int] consider
    cdef vector[int] stack
    cdef int current
    cdef set neighbours, neighbour_neighbours
    cdef int init_point, point, member

   # Number of points
    n = len(neighbourhoods)

    # Initialise labels
    labels = [0 for _ in range(n)]

    # Track assigment
    consider = [1 for _ in range(n)]

    # Start with first cluster (0 = noise)
    current = 1

    for init_point in range(n):
        # Loop over points
        if consider[init_point] == 0:
            # Point already assigned
            continue

        neighbours = neighbourhoods[init_point]
        m = len(neighbours)
        if m <= cnn_cutoff:
            # Point can not fulfill cnn condition
            labels[init_point] = 0             # Assign cluster label
            consider[init_point] = 0           # Mark point as included
            continue

        labels[init_point] = current          # Assign cluster label
        consider[init_point] = 0              # Mark point as included

        point = init_point
        while True:
            # Loop over neighbouring points
            for member in neighbours:
                if consider[member] == 0:
                    # Point already assigned
                    continue

                neighbour_neighbours = neighbourhoods[member]
                k = len(neighbour_neighbours)
                if k <= cnn_cutoff:
                    labels[member] = 0
                    consider[member] = 0
                    continue

                # conditional growth
                # if (len(neighbours.intersection(neighbourhoods[member]))
                #        >= cnn_cutoff):
                if check_similarity_set(
                        neighbours, neighbour_neighbours, cnn_cutoff):
                    labels[member] = current
                    consider[member] = 0
                    stack.push_back(member)

            if stack.size() == 0:
                # No points left to check
                break

            while stack.size() > 0:
                point = stack.back()  # Get the next point from the queue
                stack.pop_back()
                neighbours = neighbourhoods[point]
                m = len(neighbours)
                if m <= cnn_cutoff:
                    # Point can not fulfill cnn condition
                    labels[init_point] = 0             # Assign cluster label
                    consider[init_point] = 0           # Mark point as included
                    continue
                break

        current += 1

    return labels


def predict_from_neighbours(
        cnn_cutoff: int,
        neighbourhoods: List[Set[int]],
        labels: Sequence[int],
        consider: Sequence[int],
        base_labels: Sequence[int],
        clusters: Collection[int]):

    """"""

    for point in range(labels.shape[0]):
        if not consider[point]:
            continue

        neighbours = neighbourhoods[point]
        for member in neighbours:
            if not base_labels[member] in clusters:
                continue

            if (len(neighbours.intersection(neighbourhoods[member]))
                    >= cnn_cutoff):
                consider[point] = 0
                labels[point] = base_labels[member]
                break

    return
