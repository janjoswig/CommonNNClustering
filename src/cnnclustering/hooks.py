import numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE
from cnnclustering import _types, _fit


COMPONENT_ALT_KW_MAP = {
    "input": "input_data",
    "data": "input_data",
    "n": "neighbours",
    "na": "neighbours",
    "nb": "neighbour_neighbours",
    "getter": "neighbours_getter",
    "ogetter": "neighbours_getter_other",
    "ngetter": "neighbours_getter",
    "ongetter": "neighbours_getter_other",
    "dgetter": "distance_getter",
    "checker": "similarity_checker",
    "q": "queue",
}

COMPONENT_KW_TYPE_ALIAS_MAP = {
    "neighbour_neighbours": "neighbours",
    "neighbour_getter_other": "neighbours_getter",
}


COMPONENT_NAME_TYPE_MAP = {
    "input_data": {
        "components_mview": _types.InputDataExtComponentsMemoryview,
        "neighbourhoods_mview": _types.InputDataExtNeighbourhoodsMemoryview
    },
    "neighbours_getter": {
        "brute_force": _types.NeighboursGetterExtBruteForce,
        "lookup": _types.NeighboursGetterExtLookup,
    },
    "distance_getter": {
        "metric": _types.DistanceGetterExtMetric,
        "lookup": _types.DistanceGetterExtLookup,
    },
    "neighbours": {
        "vector": _types.NeighboursExtVector,
        "uset": _types.NeighboursExtCPPUnorderedSet,
        "vuset": _types.NeighboursExtVectorCPPUnorderedSet,
    },
    "metric": {
        "dummy": _types.MetricExtDummy,
        "precomputed": _types.MetricExtPrecomputed,
        "euclidean": _types.MetricExtEuclidean,
        "euclidean_r": _types.MetricExtEuclideanReduced,
        "euclidean_periodic_r": _types.MetricExtEuclideanPeriodicReduced,
    },
    "similarity_checker": {
        "contains": _types.SimilarityCheckerExtContains,
        "switch": _types.SimilarityCheckerExtSwitchContains,
        "screen": _types.SimilarityCheckerExtScreensorted,
    },
    "queue": {
        "fifo": _types.QueueExtFIFOQueue
    },
    "fitter": {
        "bfs": _fit.FitterExtBFS
    }
}


def get_registered_recipe(key):
    registered_recipes = {
        "none": {},
        "points": {
            "input_data": "components_mview",
            "fitter": "bfs",
            "fitter.ngetter": "brute_force",
            "fitter.na": "vuset",
            "fitter.checker": "switch",
            "fitter.queue": "fifo",
            "fitter.ngetter.dgetter": "metric",
            "fitter.ngetter.dgetter.metric": "euclidean_r",
        },
        "distances": {
            "input_data": "components_mview",
            "fitter": "bfs",
            "fitter.ngetter": "brute_force",
            "fitter.na": "vuset",
            "fitter.checker": "switch",
            "fitter.queue": "fifo",
            "fitter.ngetter.dgetter": "metric",
            "fitter.ngetter.dgetter.metric": "precomputed",
        },
        "neighbourhoods": {
            "input_data": "neighbourhoods_mview",
            "fitter": "bfs",
            "fitter.ngetter": "lookup",
            "fitter.na": "vuset",
            "fitter.checker": "switch",
            "fitter.queue": "fifo",
        },
        "sorted_neighbourhoods": {
            "input_data": "neighbourhoods_mview",
            "fitter": "bfs",
            "fitter.ngetter": ("lookup", (), {"is_sorted": True}),
            "fitter.na": "vector",
            "fitter.checker": "screen",
            "fitter.queue": "fifo",
        }
    }

    return registered_recipes[key.lower()]


def prepare_pass(data):
    """Dummy preparation hook

    Use if no preparation of input data is desired.

    Args:
        data: Input data that should be prepared.

    Returns:
        (data,), {}
    """

    return (data,), {}


def prepare_points_from_parts(data):
    r"""Prepare input data points

    Use when point components are passed as sequence of parts, e.g. as

        >>> input_data, meta = prepare_points_parts([[[0, 0],
        ...                                           [1, 1]],
        ...                                          [[2, 2],
        ...                                           [3,3]]])
        >>> input_data
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3]])

        >>> meta
        {"edges": [2, 2]}

    Recognised data formats are:

        * Sequence of length *d*:
            interpreted as 1 point with *d* components.
        * 2D Sequence (sequence of sequences all of same length) with
            length *n* (rows) and width *d* (columns):
            interpreted as *n* points with *d* components.
        * Sequence of 2D sequences all of same width:
            interpreted as parts (groups) of points.

    The returned input data format is compatible with:

        * `cnnclustering._types.InputDataExtPointsMemoryview`

    Args:
        data: Input data that should be prepared.

    Returns:
        * Formatted input data (NumPy array of shape
            :math:`\sum n_\mathrm{part}, d`)
        * Dictionary of meta-information

    Notes:
        Does not catch deeper nested formats.
    """

    try:
        d1 = len(data)
    except TypeError as error:
        raise error

    finished = False

    if d1 == 0:
        # Empty sequence
        data = [np.array([[]])]
        finished = True

    if not finished:
        try:
            d2 = [len(x) for x in data]
            all_d2_equal = (len(set(d2)) == 1)
        except TypeError:
            # 1D Sequence
            data = [np.array([data])]
            finished = True

    if not finished:
        try:
            d3 = [len(y) for x in data for y in x]
            all_d3_equal = (len(set(d3)) == 1)
        except TypeError:
            if not all_d2_equal:
                raise ValueError(
                    "Dimension mismatch"
                )
            # 2D Sequence of sequences of same length
            data = [np.asarray(data)]
            finished = True

    if not finished:
        if not all_d3_equal:
            raise ValueError(
                "Dimension mismatch"
            )
        # Sequence of 2D sequences of same width
        data = [np.asarray(x) for x in data]
        finished = True

    meta = {}

    meta["edges"] = [x.shape[0] for x in data]

    data_args = (np.asarray(np.vstack(data), order="C", dtype=P_AVALUE),)
    data_kwargs = {"meta": meta}

    return data_args, data_kwargs


def prepare_neighbourhoods(data):

    n_neighbours = [len(s) for s in data]
    pad_to = max(n_neighbours)

    data = [
        np.pad(a, (0, pad_to - n_neighbours[i]), mode="constant", constant_values=0)
        for i, a in enumerate(data)
        ]

    meta = {}

    data_args = (
        np.asarray(data, order="C", dtype=P_AINDEX),
        np.asarray(n_neighbours, dtype=P_AINDEX)
        )

    data_kwargs = {"meta": meta}

    return data_args, data_kwargs
