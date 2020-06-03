from collections import deque
from typing import List, Set
from typing import Sequence, Collection

import numpy as np


def fit_from_PointsArray(
        points,
        labels,
        radius_cutoff: float,
        cnn_cutoff: int):
    """Apply CNN clustering from array of points

    """

    n = points.shape[0]
    d = points.shape[1]

    current = 1
    membercount = 1  # Optional for min. clustersize
    q = deque()  # Queue

    # visited  # TODO Move out of this function
    consider = np.ones(n, dtype=np.uint8)

    radius_cutoff = radius_cutoff**2
    # for distance squared

    for init_point in range(n):
        if consider[init_point] == 0:
            continue
        consider[init_point] = 0

        neighbours = get_neighbours_PointsArray(
            init_point, points,
            n, d,
            radius_cutoff
            )

        if len(neighbours) <= cnn_cutoff:
            continue

        labels[init_point] = current
        membercount = 1
        point = init_point

        while True:
            for member in neighbours:
                if consider[member] == 0:
                    continue

                neighbour_neighbours = get_neighbours_PointsArray(
                    member, points,
                    n, d,
                    radius_cutoff
                    )

                if len(neighbour_neighbours) <= cnn_cutoff:
                    consider[member] = 0
                    continue

                if check_similarity_set(
                        neighbours, neighbour_neighbours, cnn_cutoff):
                    consider[member] = 0
                    labels[member] = current
                    membercount += 1
                    q.append(member)

            if not q:
                if membercount == 1:
                    # Revert cluster assignment
                    labels[init_point] = 0
                    current -= 1
                break

            point = q.popleft()
            neighbours = get_neighbours_PointsArray(
                point, points,
                n, d,
                radius_cutoff
                )

        current += 1


def get_neighbours_PointsArray(
        point,
        points,
        n, dim,
        radius_cutoff):
    """Caculate neighbours of a point"""

    neighbours = set()
    p = points[point]

    for i in range(n):
        if i == point:
            # No self counting
            continue

        r = get_distance_squared_euclidean_PointsArray(
                p, points[i], dim
                )

        if r <= radius_cutoff:
            neighbours.add(i)

    return neighbours


def get_distance_squared_euclidean_PointsArray(
        a, b, dim):
    """Calculate squared euclidean distance between points (parallel)

    Args:
        a, b: Point container supporting the buffer protocol. `a`
            and `b` have to be of length >= `dim`.
        dim: Dimensions to consider.
    """

    total = 0

    for i in range(dim):
        total += (a[i] - b[i])**2

    return total


def check_similarity_set(a: Set[int], b: Set[int], c: int) -> bool:
    """Check if similarity criterion is fulfilled.

    Args:
        a: Sequence of point indices
        b: Sequence of point indices
        c: Similarity cut-off

    Returns:
        True if set `a` and set `b` have at least `c` common
        elements
    """

    if len(a.intersection(b)) >= c:
        return True
    return False


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
