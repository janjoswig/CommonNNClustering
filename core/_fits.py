from collections import deque
from typing import List, Set

import numpy as np


def fit_from_neighbours(
        cnn_cutoff: int,
        neighbours: List[Set[int]]) -> List[int]:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library only.  Minor design difference to
    `fit_stdlib_from_neighbours_index` in looping over all points
    instead of checking for `include.index(True)`. Expected to be faster
    when many cluster initialisations are done. Also, checking queue
    at the end of the inner loop instead of in the begenning (no
    performance difference expected, but unnecessary adding of initial
    points to queue is avoided).

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: List of length #points containing sets of
            neighbouring point indices

    Returns:
        Labels
    """

    len_ = len(neighbours)

    # Initialise labels
    labels = [0 for _ in range(len_)]

    # Track assigment
    include = [True for _ in range(len_)]

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    for point in range(len_):
        if not include[point]:
            # Point already assigned
            continue
        labels[point] = current          # Assign cluster label
        include[point] = False           # Mark point as included

        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            for member in neigh:
                if not include[member]:
                    # Point already assigned
                    continue

                # conditional growth
                if len(neigh.intersection(neighbours[member])) >= cnn_cutoff:
                    labels[member] = current
                    include[member] = False
                    queue.append(member)

            if not queue:
                break
            point = queue.popleft()  # FIFO

        current += 1

    return labels


def get_neighbours_brute_array(self, point: int, r: float) -> Set[int]:
    """Compute neighbours of a point by (squared) distance

    Args:
        point: Point index
        r: Distance cut-off

    Returns:
        Array of indices of points that are neighbours
        of `point` within `r`.
    """

    r = r**2
    return np.where(np.sum((self.data.data - point)**2, axis=1) < r)[0]