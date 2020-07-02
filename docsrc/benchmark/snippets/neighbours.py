import numpy as np


def get_neighbours_from_dist_baseline(dist, radius_cutoff):
    """Original implementation of neigbourlist computing

    Published in [...] For illustration purposes variables have been
    sometimes renamed, PEP8 conformance was loosely ensured and
    docstrings and more comments where included.  To make new comments
    distinguishable from the original ones a double number sign (##) was
    used.

    Compute complete neighbour list from (dense) distance matrix

    Args:
       dist (numpy.ndarray): Distance matrix of shape
          (#points, #points).
    Returns:
       Neighbourlist as list of length #points of numpy.ndarrays of
       neighbouring point indices.
       Neighbourcount as numpy.ndarray of length #points
    """

    ## Number of points
    n = np.int_(np.shape(dist)[0])  ## n = dist.shape[0]

    # Prepare neighborlist
    n_neighbours = np.zeros(n)  ## Neighbour count for each point
    neighbours = [0] * n        ## Neighbourlist container

    # Calculate neighborlist
    for i in range(n):
        neigh = np.nonzero(dist[i, :] <= radius_cutoff)[0]
        ## Neighbours of a point i

        n_neighbours[i] = len(neigh)
        neighbours[i] = neigh

    return neighbours, n_neighbours