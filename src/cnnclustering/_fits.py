"""Common-nearest-neighbour clustering functionality

Replaced by Cython extension module `_cfits` in production and
only used for demonstration, exploration, testing, and compatibility.
"""

from collections import deque
from typing import Type
from typing import List, Set
from typing import Sequence, Collection

import numpy as np


def fit(
        input_data,
        labels,
        consider,
        radius_cutoff,
        cnn_cutoff,
        ):
    """Generic clustering"""

    n = input_data.shape[0]

    current = 1
    q = deque()      # Queue

    for init_point in range(n):
        if consider[init_point] == 0:
            continue
        consider[init_point] = 0

        neighbours = input_data.get_neighbours(
            init_point,
            )

        if not neighbours.enough():
            continue

        labels[init_point] = current
        point = init_point

        while True:
            for member in neighbours:
                if consider[member] == 0:
                    continue

                neighbour_neighbours = input_data.get_neighbours(
                    member
                    )

                if not neighbour_neighbours.enough():
                    consider[member] = 0
                    continue

                if neighbours.check_similarity(
                        neighbour_neighbours):
                    consider[member] = 0
                    labels[member] = current
                    q.append(member)

            if not q:
                break

            point = q.popleft()
            neighbours = input_data.get_neighbours(
                point
                )

        current += 1


def fit_from_PointsArray(
        points: Type[np.ndarray],
        labels: Type[np.ndarray],
        consider: Type[np.ndarray],
        radius_cutoff: float,
        cnn_cutoff: int):
    """Apply common-nearest-neighbour clustering

    Starting from NumPy array of points
    """

    n = points.shape[0]
    d = points.shape[1]

    current = 1
    membercount = 1  # Optional for min. clustersize
    q = deque()      # Queue

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


def check_similarity_set(a: Set[int], b: Collection[int], c: int) -> bool:
    """Check if similarity criterion is fulfilled.

    Args:
        a: Set of point indices
        b: Collection of point indices
        c: Similarity cut-off

    Returns:
        True if set `a` and set `b` have at least `c` common
        elements
    """

    if len(a.intersection(b)) >= c:
        return True
    return False


def check_similarity_collection(
        a: Collection[int], b: Collection[int], c: int) -> bool:
    """Check if similarity criterion is fulfilled.

    Args:
        a: Collection of point indices
        b: Collection of point indices
        c: Similarity cut-off

    Returns:
        True if set `a` and set `b` have at least `c` common
        elements
    """

    if len(set(a).intersection(b)) >= c:
        return True
    return False


def check_similarity_array(
        a: Type[np.ndarray], b: Type[np.ndarray], c: int) -> bool:
    """Check if similarity criterion is fulfilled.

    Args:
        a: NumPy array of point indices
        b: NumPy array of point indices
        c: Similarity cut-off

    Returns:
        True if set `a` and set `b` have at least `c` common
        elements
    """

    if np.intersect1d(a, b, assume_unique=True).shape[0] >= c:
        return True
    return False


def fit_from_NeighbourhoodsList(
        neighbourhoods: List[Set[int]],
        labels: Type[np.ndarray],
        consider: Type[np.ndarray],
        cnn_cutoff: int,
        self_counting: bool) -> List[int]:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbourhoods.

    Args:
        neighbourhoods: List of length #points containing sets of
            neighbouring point indices
        labels: Output data container supporting the buffer protocoll
            for cluster label assignments.  Needs to be of same length
            as neighbourhoods.
        consider: Data container supporting the buffer protocoll of
            same length as labels.  Elements should be either `False`
             or `True`
            indicating if the point with the corresponding index should
            be included for the clustering.
        cnn_cutoff: Points need to share at least this many neighbours
            to be assigned to the same cluster (similarity criterion).
        self_counting: If `True`, accounts for points being in their
            own neighbours and modifies `cnn_cutoff`.

    Returns:
        None
    """

    # Account for self-counting in neighbourhoods
    if self_counting:
        cnn_cutoff += 1
        cnn_cutoff_ = cnn_cutoff + 1
    else:
        cnn_cutoff_ = cnn_cutoff

    # Number of points
    n = labels.shape[0]

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    for init_point in range(n):
        if not consider[init_point]:
            # Point already assigned
            continue
        consider[init_point] = False           # Mark point as included

        neighbours = neighbourhoods[init_point]
        if len(neighbours) <= cnn_cutoff:
            # Point can not fulfill cnn condition
            continue

        labels[init_point] = current           # Assign cluster label
        membercount = 1

        while True:
            # Loop over neighbouring points
            for member in neighbours:
                if not consider[member]:
                    # Point already assigned
                    continue

                neighbour_neighbours = neighbourhoods[member]
                if len(neighbour_neighbours) <= cnn_cutoff:
                    consider[member] = False
                    continue

                # conditional growth
                if check_similarity_set(
                        neighbours, neighbour_neighbours, cnn_cutoff):
                    labels[member] = current
                    consider[member] = False
                    membercount += 1
                    queue.append(member)

            if not queue:
                if membercount == 1:
                    # Revert cluster assignment
                    labels[init_point] = 0
                    current -= 1
                break

            point = queue.popleft()  # FIFO

            neighbours = neighbourhoods[point]

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
