from collections import deque
from typing import List, Set
from typing import Sequence, Collection

import numpy as np


def fit_from_neighbours(
        cnn_cutoff: int,
        neighbourhoods: List[Set[int]]) -> List[int]:
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

    # Number of points
    len_ = len(neighbourhoods)

    # Initialise labels
    labels = [0 for _ in range(len_)]

    # Track assignment
    consider = [True for _ in range(len_)]

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    for init_point in range(len_):
        if not consider[init_point]:
            # Point already assigned
            continue

        neighbours = neighbourhoods[init_point]
        if len(neighbours) <= cnn_cutoff:
            # Point can not fulfill cnn condition
            labels[init_point] = 0             # Assign cluster label
            consider[init_point] = False       # Mark point as included
            continue

        labels[init_point] = current           # Assign cluster label
        consider[init_point] = False           # Mark point as included

        point = init_point
        while True:
            # Loop over neighbouring points
            for member in neighbours:
                if not consider[member]:
                    # Point already assigned
                    continue

                neighbour_neighbours = neighbourhoods[member]
                if len(neighbour_neighbours) <= cnn_cutoff:
                    labels[member] = 0
                    consider[member] = False
                    continue

                # conditional growth
                if (len(neighbours.intersection(neighbour_neighbours))
                        >= cnn_cutoff):
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

            try:
                while True:
                    point = queue.popleft()  # FIFO
                    neighbours = neighbourhoods[point]
                    if len(neighbourhoods[point]) <= cnn_cutoff:
                        labels[point] = 0
                        consider[point] = False
                        continue
                    break
            except IndexError:
                # Queue empty
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
    """Predict cluster labels on the basis of previously assigned labels

    Args:
        cnn_cutoff: Points need to share at least this many neighbours
            to be assigned to the same cluster (similarity criterion).
        neighbourhoods: List of length #points (in the data set for
            which labels should be predicted) containing sets of
            neighbouring point indices (in the data set for which
            labels have been previously assigned).
        labels: Sequence of length #points for the predicted
            assignments.
        consider: Sequence of length #points indicating for which points
            a prediction should be attempted.
        base_labels: Sequence of length #points\' for which labels have
            been.
        clusters: Collection (ideally a set) of cluster labels for
            which an assignment should be attempted.
    """

    # for point in range(len(labels))
    for point in range(labels.size):
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
    return set(np.where(np.sum((self.data.data - point)**2, axis=1) < r)[0])
