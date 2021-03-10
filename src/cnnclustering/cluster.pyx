import numpy as np
cimport numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from cnnclustering._types import (
    InputDataNeighboursSequence,
    InputDataExtPointsMemoryview,
    NeighboursGetterBruteForce,
    NeighboursGetterLookup,
    NeighboursGetterExtBruteForce,
    NeighboursList,
    NeighboursSet,
    NeighboursExtVector,
    MetricDummy,
    MetricEuclidean,
    MetricExtDummy,
    MetricExtEuclidean,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    QueueFIFODeque,
    QueueExtFIFOQueue,
)
from cnnclustering._fit import FitterExtBFS, FitterBFS


COMPONENT_NAME_TYPE_MAP = {
    "input_data": {
        "array2d": InputDataExtPointsMemoryview,
    },
    "neighbours_getter": {
        "brute_force": NeighboursGetterExtBruteForce
    },
    "neighbours": {
        "vector": NeighboursExtVector,
    },
    "metric": {
        "euclidean": MetricExtEuclidean,
    },
    "similarity_checker": {
        "contains": SimilarityCheckerExtContains,
    },
    "queue": {
        "fifo": QueueExtFIFOQueue
    },
    "fitter": {
        "bfs": FitterExtBFS
    }
}

COMPONENT_KW_ALT = {
    "neighbour_neighbours": "neighbours",
}

def prepare_clustering(input_data, preparation_hook=None, **recipe):
    """Initialise clustering with input data"""


    default_recipe = recipe = {
        "input_data": "array2d",
        "neighbours_getter": "brute_force",
        "neighbours": ("vector", (10,), {}),
        "neighbour_neighbours": ("vector", (10,), {}),
        "metric": "euclidean",
        "similarity_checker": "contains",
        "queue": "fifo",
        "fitter": "bfs",
    }

    default_recipe.update(recipe)

    if preparation_hook is not None:
        input_data = preparation_hook(input_data)
    else:
        input_data = np.asarray(input_data)

    components = {}
    for component_kw, component_details in default_recipe.items():
        args = ()
        kwargs = {}
        component_type = None

        _component_kw = COMPONENT_KW_ALT.get(
            component_kw, component_kw
            )

        if isinstance(component_details, str):
            component_type = COMPONENT_NAME_TYPE_MAP[_component_kw][
                component_details
                ]

        elif isinstance(component_details, tuple):
            component_type, args, kwargs = component_details
            if isinstance(component_type, str):
                component_type = COMPONENT_NAME_TYPE_MAP[_component_kw][
                    component_type
                    ]

        else:
            component_type = component_details

        if _component_kw == "input_data":
            args = (input_data, *args)

        if component_type is not None:
            components[component_kw] = component_type(
                *args, **kwargs
            )

    return Clustering(**components)


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

    @property
    def labels(self):
        return self._labels
