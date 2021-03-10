import numpy as np
cimport numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


class Clustering:

    def __init__(
            self,
            input_data=None,
            neighbours_getter=None,
            neighbours=None,
            neighbour_neighbours=None,
            metric=None,
            similarity_checker=None,
            queue=None,
            fitter=None,
            labels=None):

        self._input_data = input_data
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._metric = metric
        self._similarity_checker = similarity_checker
        self._queue = queue
        self._fitter = fitter
        self._labels = labels

    def fit(self, radius_cutoff: float, cnn_cutoff: int) -> None:

        cdef ClusterParameters cluster_params = ClusterParameters(
            radius_cutoff, cnn_cutoff
            )

        self._labels = Labels(
            np.zeros(self._input_data.n_points, order="c", dtype=P_AINDEX)
            )

        self._fitter.fit(
            self._input_data,
            self._neighbours_getter,
            self._neighbours,
            self._neighbour_neighbours,
            self._metric,
            self._similarity_checker,
            self._queue,
            self._labels,
            cluster_params
            )

        return
