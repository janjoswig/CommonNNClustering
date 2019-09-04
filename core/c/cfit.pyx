from sortedcontainers import SortedList
from collections import defaultdict
import numpy as np

def fit(dist, float radius_cutoff, int cnn_cutoff, int member_cutoff,
    int max_clusters):

    n_points = len(dist)
    # calculate neighbour list
    neighbours = np.asarray([
        np.where((x > 0) & (x <= radius_cutoff))[0]
        for x in dist
        ])
    n_neighbours = np.asarray([len(x) for x in neighbours])
    incl = np.ones(len(neighbours), dtype=bool)
    incl[np.where(n_neighbours < cnn_cutoff)[0]] = False
    
    _clusterdict = defaultdict(SortedList)
    _clusterdict[0].update(np.where(incl == False)[0])
    _labels = np.zeros(n_points).astype(int)
    current = 1

    # print(f"Initialisation done: {time.time() - go}")

    enough = False
    while any(incl) and not enough:
        # find point with highest neighbour count
        point = np.where(
            (n_neighbours == np.max(n_neighbours[incl]))
            & (incl == True)
        )[0][0]
        _clusterdict[current].add(point)
        new_point_added = True
        _labels[point] = current
        incl[point] = False
        # print(f"Opened cluster {current}: {time.time() - go}")
        # done = 0
        while new_point_added:
            new_point_added = False
            # for member in _clusterdict[current][done:]:
            for member in [
                added_point for added_point in _clusterdict[current]
                if any(incl[neighbours[added_point]])
                ]:
                # Is the SortedList dangerous here?
                for neighbour in neighbours[member][incl[neighbours[member]]]:
                    common_neighbours = (
                        set(neighbours[member])
                        & set(neighbours[neighbour])
                        )

                    if len(common_neighbours) >= cnn_cutoff:
                    # and (point in neighbours[neighbour])
                    # and (neighbour in neighbours[point]):
                        _clusterdict[current].add(neighbour)
                        new_point_added = True
                        _labels[neighbour] = current
                        incl[neighbour] = False

            # done += 1   
        current += 1

        if max_clusters is not None:
            if current == max_clusters+1:
                enough = True