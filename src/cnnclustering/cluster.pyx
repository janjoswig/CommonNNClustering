import numpy as np
cimport numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from cnnclustering._fit import Fitter

class Clustering:

    def __init__(
            self,
            input_data=None,
            neighbours_getter=None,
            metric=None,
            fitter=None):

        self._input_data = input_data
        self._neighbours_getter = neighbours_getter
        self._metric = metric
        self._fitter = fitter
        self._labels = np.zeros(5, dtype=P_AINDEX)
        self._consider = np.ones_like(self._labels, dtype=P_ABOOL)

    def _fit(
        self,
        INPUT_DATA input_data,
        NEIGHBOURS_GETTER neighbours_getter,
        NEIGHBOURS special_dummy,
        METRIC metric,
        FITTER fitter,
        AVALUE radius_cutoff,
        AINDEX cnn_cutoff):

        cdef ClusterParameters cluster_params = ClusterParameters(
            radius_cutoff, cnn_cutoff
            )
        cdef ClusterParameters *cluster_params_ptr = &cluster_params

        cdef AINDEX[::1] labels = self._labels
        cdef ABOOL[::1] consider = self._consider

        if FITTER is object:
            fitter.fit(
                input_data,
                neighbours_getter,
                special_dummy,
                metric,
                labels,
                consider,
                cluster_params,
                )
        else:
            fitter.fit(
                input_data,
                neighbours_getter,
                special_dummy,
                metric,
                &labels[0],
                &consider[0],
                cluster_params_ptr,
                )


    def fit(self, radius_cutoff: float, cnn_cutoff: int) -> None:


        self._fit(
            self._input_data,
            self._neighbours_getter,
            self._neighbours_getter.neighbours_dummy,
            self._metric,
            self._fitter,
            radius_cutoff,
            cnn_cutoff,
            )

        return