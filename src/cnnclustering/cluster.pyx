import numpy as np
cimport numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


class Clustering:

    def __init__(
            self,
            input_data=None,
            neighbours_getter=None,
            metric=None,
            similarity_checker=None,
            queue=None,
            fitter=None,
            labels=None):

        self._input_data = input_data
        self._neighbours_getter = neighbours_getter
        self._metric = metric
        self._similarity_checker = similarity_checker
        self._queue = queue
        self._fitter = fitter
        self._labels = labels

    def _fit(
        self,
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        SIMILARITY_CHECKER similarity_checker,
        QUEUE queue,
        FITTER fitter,
        Labels labels,
        ClusterParameters cluster_params):

        if (FITTER is object) or (
                (INPUT_DATA in INPUT_DATA_EXT) and
                (NEIGHBOURS_GETTER in NEIGHBOURS_GETTER_EXT) and
                (NEIGHBOURS in NEIGHBOURS_EXT) and
                (METRIC in METRIC_EXT) and
                (SIMILARITY_CHECKER in SIMILARITY_CHECKER_EXT) and
                (QUEUE in QUEUE_EXT)):
            fitter.fit(
                input_data,
                neighbours_getter,
                special_dummy,
                metric,
                similarity_checker,
                queue,
                labels,
                cluster_params,
                )
        else:
            raise TypeError(
                ""
                )

    def fit(self, radius_cutoff: float, cnn_cutoff: int) -> None:

        cdef ClusterParameters cluster_params = ClusterParameters(
            radius_cutoff, cnn_cutoff
            )

        self._labels = Labels(
            np.arange(self._input_data.n_points, dtype=P_AINDEX)
            )

        self._fit(
            self._input_data,
            self._neighbours_getter,
            self._neighbours_getter.neighbours_dummy,
            self._metric,
            self._similarity_checker,
            self._queue,
            self._fitter,
            self._labels,
            cluster_params
            )

        return
