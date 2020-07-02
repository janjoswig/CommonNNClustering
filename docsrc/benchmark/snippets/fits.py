from collections import deque, defaultdict
from typing import List, Set

import numpy as np


def check_similarity_set(a: Set, b: Set, c: int) -> bool:
    """Constrained intersection between to sets

    Test if two sets `a` and `b` share a number of at least `c` common
    elements.

    In pure Python this is actually slower than using built-in set
    intersections (`a & b`) which is an optimised Python/C API
    operation, especially when this function is not used inline.
    This function only serves as a blue-print for the
    cythonised equivalent in `_cfits`.
    """

    # Sets to compare need to have at least c members
    if (len(a) < c) or (len(b) < c):
        return False

    common = 0  # Common element counter
    for x in a:
        # iterate over elements in set a
        if x in b:
            common += 1
        if common == c:
            return True
    return False


def fit_from_neighbours_baseline(
        cnn_cutoff, neighbours):
    """Original implementation of CNN clustering

    Published in [...] For illustration purposes variables have been
    sometimes renamed, PEP8 conformance was loosely ensured and
    docstrings and more comments where included.  To make new comments
    distinguishable from the original ones a double number sign (# #)
    was used.

    Assign cluster labels to points starting from pre-computed neighbour
    list.

    Args:
        neighbours: Neighbourlist as collection of collections
        cnn_cutoff: Similarity criterion
    """

    def ismember(a, b):
        """Test if two iterables have common elements

        Args:
           a, b (iterable): Iterables to compare
        Returns:
           List of same length as `a` in which the elements
           indicate if an element of `a` is in `b`, where
              - the list element is -1 if the element of `a` is not
                 in `b`
              - the list element is the index of the element in ` b`
                 if the element of `a` is in `b`
        """

        bi = {}
        # # Container for unique elements in b {element: index_in_b}
        for i, el in enumerate(b):
            if el not in bi:
                bi[el] = i
        return [bi.get(itm, -1) for itm in a]

    n = len(neighbours)

    n_neighbours = np.zeros(n)  # # Neighbour count for each point
    for i in range(n):
        n_neighbours[i] = len(neighbours[i])

    # Preset parameters for the clustering

    cn = 0                # Cluster Number
    w = 0                 # Control value I
    # # Indicates clustering done, 0: False, >0: True

    clusters = [0] * n    # Preset cluster list
    # # Used instead of labels

    # Filter noise points
    n_neighbours[n_neighbours <= cnn_cutoff] = 0  # # No (0) neighbours
    # # Remove for comparability.
    # # Filtering should be done before the actual fit

    while w < 1:  # # While not finished
        # Find maximal number of nearest neighbors
        nmax = np.nonzero(n_neighbours == max(n_neighbours))[0]
        # # Remove for comparability.
        # # Initial cluster points can be choosen arbitrarily.
        # # Only interesting when clusters should be found approximately
        # # in the order of size, which is usefull when something like
        # # max_clusters is set.

        # Reset cluster
        C = np.zeros(n, dtype=int)

        # Write point with the highest density into the cluster
        C[0] = nmax[0]

        # Cluster index
        cl = int(1)  ## How many points are in the current cluster

        # Control value II
        cc = 1
        # # New point added? 0: False, >0: True

        # Cluster index for new added points
        lv = 0
        ci = np.zeros(n, dtype=int)

        # Mask point with the highest density
        n_neighbours[nmax[0]] = 0

        while cc > 0:
            # Reset control value II and define new limits
            cc = 0
            ci[lv] = cl

            # # Loop over points added to the cluster
            for a in C[ci[lv - 1]: ci[lv]]:

                # Extract Neighborlist of a within radius_cutoff
                Na = neighbours[a]

                # Compare Neighborlist of a and all reachable datapoints
                for b in Na:
                    # Check if point is already clustered
                    if n_neighbours[b] > 0:
                        Nb = neighbours[b]

                        tcc = np.asarray(ismember(Na, Nb))
                        tc = len(tcc[tcc >= 0])

                        # Check if b in the Nearest Neighbors of a
                        if b in Na:
                            tb = 1
                        else:
                            tb = 0

                        # Check if a in the Nearest Neighbors of b
                        if a in Nb:
                            ta = 1
                        else:
                            ta = 0

                        # Check truncation criterion
                        if tc >= cnn_cutoff and ta > 0 and tb > 0:
                            # Add point to the cluster
                            C[cl] = b
                            cl = cl + 1
                            cc = cc + 1

                            # Mask clustered point
                            n_neighbours[b] = 0

            # Update lv
            lv = lv + 1

        # Write cluster to the cluster list
        Cc = np.int_(C[:cl])
        clusters[cn] = Cc
        cn = cn + 1  # # Cluster count increased

        # Truncation criteria
        if sum(n_neighbours) == 0:  # All particles are clustered
            w = 1
        if cn == n:  # The maximal number of clusters is reached
            w = 2

    return clusters


def fit_stdlib_from_neighbours_index(
        cnn_cutoff: int,
        neighbours: List[Set[int]]) -> List:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library only.  Uses builtin function `index`
    to find unassigned initial cluster points, which is expected to be
    fast when done seldom (compare to fit_stdlib_from_neighbours_loop).

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: List of length #points containing sets of
            neighbouring point indices

    Returns:
        Labels
    """

    # Initialise labels
    labels = [0 for _ in range(len(neighbours))]

    # Track assigment
    consider = [True for _ in range(len(neighbours))]

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    while True:
        try:
            point = consider.index(True)  # Pick starting point
        except ValueError:               # All points assigned
            break
        labels[point] = current          # Assign cluster label
        consider[point] = False           # Mark point as included
        queue.append(point)              # Add point to queue

        while queue:
            point = queue.popleft()      # FIFO

            # Loop over neighbouring points
            neigh = neighbours[point]
            for member in neigh:
                if not consider[member]:
                    # Point already assigned
                    continue

                # conditional growth
                if len(neigh.intersection(neighbours[member])) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

        current += 1

    return labels


def fit_stdlib_from_neighbours_loop(
        cnn_cutoff: int,
        neighbours: List[Set[int]]) -> List:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library only.  Minor design difference to
    `fit_stdlib_from_neighbours_index` in looping over all points
    instead of checking for `consider.index(True)`. Expected to be faster
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
    consider = [True for _ in range(len_)]

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    for point in range(len_):
        if not consider[point]:
            # Point already assigned
            continue
        labels[point] = current          # Assign cluster label
        consider[point] = False           # Mark point as included

        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            for member in neigh:
                if not consider[member]:
                    # Point already assigned
                    continue

                # conditional growth
                if len(neigh.intersection(neighbours[member])) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

            if not queue:
                break
            point = queue.popleft()  # FIFO

        current += 1

    return labels


def fit_stdlib_from_neighbours_loop_membercheck(
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
        if len(neighbours) < cnn_cutoff:
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
                if len(neighbour_neighbours) < cnn_cutoff:
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
                    if len(neighbourhoods[point]) < cnn_cutoff:
                        labels[point] = 0
                        consider[point] = False
                        continue
                    break
            except IndexError:
                # Queue empty
                break

        current += 1

    return labels


def fit_numpy_mix(cnn_cutoff, neighbourhoods):
    """Fit data when neighbour list has been already calculated

    Args:
    """

    len_ = neighbourhoods.shape[0]

    # Include array keeps track of assigned points
    consider = np.ones(len_, dtype=bool)

    # Exclude all points with less than n neighbours
    n_neighbours = np.asarray([x.size for x in neighbourhoods])
    consider[np.where(n_neighbours < cnn_cutoff)[0]] = False

    _clusterdict = defaultdict(set)
    _clusterdict[0].update(np.where(~consider)[0])
    _labels = np.zeros(len_, dtype=int)
    current = 1

    enough = False
    while any(consider) and not enough:
        # find point with currently highest neighbour count
        point = np.where(
            (n_neighbours == np.max(n_neighbours[consider]))
            & consider
            )[0][0]

        _clusterdict[current].add(point)
        new_point_added = True
        _labels[point] = current
        consider[point] = False

        while new_point_added:
            new_point_added = False

            for member in [
                    added_point
                    for added_point in _clusterdict[current]
                    if any(consider[neighbourhoods[added_point]])
                    ]:

                for neighbour in neighbourhoods[member][consider[neighbourhoods[member]]]:
                    common_neighbours = (
                        set(neighbourhoods[member])
                        & set(neighbourhoods[neighbour])
                        )

                    if len(common_neighbours) >= cnn_cutoff:
                        # and (point in neighbours[neighbour])
                        # and (neighbour in neighbours[point]):
                        _clusterdict[current].add(neighbour)
                        new_point_added = True
                        _labels[neighbour] = current
                        consider[neighbour] = False

        current += 1

    return _labels


def fit_numpy_from_neighbours_index(
        cnn_cutoff: int,
        neighbours: np.ndarray) -> np.ndarray:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library and NumPy where appropriate.  Analouge
    to `fit_std_from_neighbours_index`.  Performance gain is, however,
    not anticipated as there is no "early return" equivalent to `index`
    in NumPy yet.

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: NumPy array of length #points containing arrays of
            neighbouring point indices

    Returns:
        Labels
    """

    len_ = neighbours.shape[0]

    # Initialise labels
    labels = np.zeros(len_)

    # Track assigment
    consider = np.ones(len_, dtype=bool)

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    while True:
        try:
            point = np.nonzero(consider)[0][0]  # Pick starting point
        except IndexError:
            break
        labels[point] = current            # Assign cluster label
        consider[point] = False             # Mark point as included

        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            for member in neigh:
                if not consider[member]:
                    # Point already assigned
                    continue

                # conditional growth
                if len(np.intersect1d(
                        neigh, neighbours[member], assume_unique=True
                        )) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

            if not queue:
                break
            point = queue.popleft()  # FIFO

        current += 1

    return labels


def fit_numpy_from_neighbours_loop(
        cnn_cutoff: int,
        neighbours: np.ndarray) -> np.ndarray:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library and NumPy where appropriate.  Analouge
    to `fit_std_from_neighbours_loop`.

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: NumPy array of length #points containing arrays of
            neighbouring point indices

    Returns:
        Labels
    """

    len_ = neighbours.shape[0]

    # Initialise labels
    labels = np.zeros(len_)

    # Track assigment
    consider = np.ones(len_, dtype=bool)

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    while True:
        try:
            point = np.nonzero(consider)[0][0]  # Pick starting point
        except IndexError:
            break
        labels[point] = current            # Assign cluster label
        consider[point] = False             # Mark point as included

        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            for member in neigh:
                if not consider[member]:
                    # Point already assigned
                    continue

                # conditional growth
                if len(np.intersect1d(
                        neigh, neighbours[member], assume_unique=True
                        )) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

            if not queue:
                break
            point = queue.popleft()  # FIFO

        current += 1

    return labels


def fit_numpy_from_neighbours_filtermembers(
        cnn_cutoff: int,
        neighbours: np.ndarray) -> np.ndarray:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library and NumPy where appropriate.  Modifies
    the loop over neighbour neighbours with a bulk check for inclusion.
    Also analouge to `fit_std_from_neighbours_loop`

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: NumPy array of length #points containing arrays of
            neighbouring point indices

    Returns:
        Labels
    """

    len_ = neighbours.shape[0]

    # Initialise labels
    labels = np.zeros(len_)

    # Track assigment
    consider = np.ones(len_, dtype=bool)

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    for point in range(len_):
        if not consider[point]:
            # Point already assigned
            continue
        labels[point] = current            # Assign cluster label
        consider[point] = False             # Mark point as included

        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            # Loop only over members not included
            for member in neigh[consider[neigh]]:
                # conditional growth
                if len(np.intersect1d(
                        neigh, neighbours[member], assume_unique=True
                        )) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

            if not queue:
                break
            point = queue.popleft()  # FIFO

        current += 1

    return labels


def fit_numpy_from_neighbours_membercheck(
        cnn_cutoff: int,
        neighbours: np.ndarray) -> np.ndarray:
    """Worker function variant applying the CNN algorithm.

    Assigns labels to points starting from pre-computed neighbour list.
    Uses Python standard library and NumPy where appropriate.  Modifies
    the loop over neighbour neighbours with a bulk check for inclusion.
    Additionally bulk checks neighbour list lengths to filter out points
    with less then `cnn_cutoff` members.
    Also analouge to `fit_std_from_neighbours_loop`.

    Args:
        cnn_cutoff: Similarity criterion
        neighbours: NumPy array of length #points containing arrays of
            neighbouring point indices

    Returns:
        Labels
    """

    len_ = neighbours.shape[0]

    # Initialise labels
    labels = np.zeros(len_)

    # Track assigment
    consider = np.ones(len_, dtype=bool)

    # Filter out points with not enough members
    consider[np.where(
        np.asarray([x.size for x in neighbours])
        < cnn_cutoff
        )[0]] = False

    # Start with first cluster (0 = noise)
    current = 1

    # Initialise queue of points to scan
    queue = deque()

    for init_point in range(len_):
        if not consider[init_point]:
            # Point already assigned
            continue
        labels[init_point] = current            # Assign cluster label
        consider[init_point] = False             # Mark point as included

        point = init_point
        while True:
            # Loop over neighbouring points
            neigh = neighbours[point]
            # Loop only over members not included
            for member in neigh[consider[neigh]]:
                # conditional growth
                if len(np.intersect1d(
                        neigh, neighbours[member], assume_unique=True
                        )) >= cnn_cutoff:
                    labels[member] = current
                    consider[member] = False
                    queue.append(member)

            if not queue:
                break
            point = queue.popleft()  # FIFO

        current += 1

    return labels
