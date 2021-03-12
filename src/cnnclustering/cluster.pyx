from collections import defaultdict
from collections.abc import MutableSequence
from typing import Any, NamedTuple
from typing import Optional
import weakref

import numpy as np
cimport numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from . import plot
    MPL_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    MPL_FOUND = False

try:
    import pandas as pd
    PANDAS_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    PANDAS_FOUND = False

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
    "getter": "neighbours_getter",
    "checker": "similarity_checker",
}


def prepare_clustering(data, preparation_hook=None, **recipe):
    """Initialise clustering with input data

    Args:
        data: Data that should be clustered in a format
            compatible with 'input_data' specified in the building
            `recipe`.  May go through `preparation_hook` to establish
            compatibility.

    Keyword args:
        preparation_hook: A function that takes `input_data` as a
            single argument and returns it (optionally) re-formatted.
            May return meta-information as a second return value (None
            otherwise).  If `None` uses :meth:`prepare_points_from_parts`.
        recipe: Building instructions for a
            :obj:`cnnclustering.cluster.Clustering` instance.
    """


    default_recipe = {
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
        data, meta = preparation_hook(data)
    else:
        data, meta = prepare_points_from_parts(data)

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
            args = (data, *args)

            meta.update(kwargs.get("meta", {}))
            kwargs["meta"] = meta

        if component_type is not None:
            components[component_kw] = component_type(
                *args, **kwargs
            )

    return Clustering(**components)


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

    return np.asarray(np.vstack(data), order="C", dtype=P_AVALUE), meta


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

        self.hierarchy_level = 0

        self._input_data = input_data
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._metric = metric
        self._similarity_checker = similarity_checker
        self._queue = queue
        self._fitter = fitter
        self._labels = labels

        self._children = None

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if not isinstance(value, Labels):
            value = Labels(value)
        self._labels = value

    @property
    def hierarchy_level(self):
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, value):
        self._hierarchy_level = int(value)

    def __repr__(self):
        return f"{type(self).__name}()"

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

    def isolate(self, purge: bool = True):
        """Split input data based on cluster labels"""

        if purge or (self._children is None):
            self._children = defaultdict(
                lambda: ClusteringChild(parent=self)
                )

        for label, indices in self.labels.mapping.items():
            # Assume indices to be sorted
            parent_indices = indices
            if self.labels._root_indices is None:
                root_indices = indices
            else:
                root_indices = self.labels._root_indices[indices]

        self._children[label].input_data = self.input_data.get_subset(indices)


    def summarize(
            self,
            ax: Optional[Any] = None,
            quantity: str = "time",
            treat_nan: Optional[Any] = None,
            ax_props: Optional[dict] = None,
            contour_props: Optional[dict] = None):
        """Generate a 2D plot of record values

        Record values ("time", "clusters", "largest", "noise") are
        plotted against cluster parameters (radius cutoff *r*
        and cnn cutoff *c*).

        Args:
            ax: Matplotlib Axes to plot on.  If `None`, a new Figure
                with Axes will be created.
            quantity: Record value to
                visualise:

                    * "time"
                    * "clusters"
                    * "largest"
                    * "noise"

            treat_nan: If not `None`, use this value to pad nan-values.
            ax_props: Used to style `ax`.
            contour_props: Passed on to contour.
        """

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if len(self._list) == 0:
            raise LookupError(
                "No cluster result records in summary"
                )

        ax_props_defaults = {
            "xlabel": "$R$",
            "ylabel": "$C$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        contour_props_defaults = {
                "cmap": mpl.cm.inferno,
            }

        if contour_props is not None:
            contour_props_defaults.update(contour_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plotted = plot.plot_summary(
            ax, self.to_DataFrame(),
            quantity=quantity,
            treat_nan=treat_nan,
            contour_props=contour_props_defaults
            )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted


class ClusteringChild(Clustering):
    """Clustering subclass.

    Increments the hierarchy level of
    the parent object when instantiated.

    Attributes:
        parent: Weak reference to parent
    """

    take_over_attrs = [
        "neighbours_getter",
        "neighbours",
        "neighbour_neighbours",
        "metric",
        "similarity_checker",
        "queue",
        "fitter",
        ]

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent = weakref.proxy(parent)

        for attr in self.take_over_attrs:
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(self.parent, attr))

        self.hierarchy_level = parent.hierarchy_level + 1


class Record(NamedTuple):
    """Cluster result container

    :obj:`cnnclustering.cluster.Record` instances can be returned by
    :meth:`cnnclustering.cluster.Clustering.fit` and
    are collected in :obj:`cnnclustering.cluster.Summary`.
    """

    points: int
    """Number of points in the clustered data set."""

    r: float
    """Radius cutoff :math:*r*."""

    c: int
    """CNN cutoff :math:*c* (similarity criterion)"""

    min: int
    """Member cutoff. Valid clusters have at least this many members."""

    max: int
    """Maximum cluster number. After sorting, only the biggest `max` clusters are kept."""

    clusters: int
    """Number of identified clusters."""

    largest: float
    """Ratio of points in the largest cluster."""

    noise: float
    """Ratio of points classified as outliers."""

    time: float
    """Measured execution time for the fit, including sorting in seconds."""


class Summary(MutableSequence):
    """List like container for cluster results

    Stores instances of :obj:`cnnclustering.cluster.Record`.
    """

    def __init__(self, iterable=None):
        if iterable is None:
            iterable = []

        self._list = []
        for i in iterable:
            self.append(i)

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, item):
        if isinstance(item, Record):
            self._list.__setitem__(key, item)
        else:
            raise TypeError(
                "Summary can only contain records of type `Record`"
            )

    def __delitem__(self, key):
        self._list.__delitem__(key)

    def __len__(self):
        return self._list.__len__()

    def __str__(self):
        return self._list.__str__()

    def insert(self, index, item):
        if isinstance(item, Record):
            self._list.insert(index, item)
        else:
            raise TypeError(
                "Summary can only contain records of type `Record`"
            )

    def to_DataFrame(self):
        """Convert list of records to (typed) :obj:`pandas.DataFrame`

        Returns:
            :obj:`pandas.DataFrame`
        """

        if not PANDAS_FOUND:
            raise ModuleNotFoundError("No module named 'pandas'")

        _record_dtypes = [
            pd.Int64Dtype(),  # points
            np.float64,       # r
            pd.Int64Dtype(),  # n
            pd.Int64Dtype(),  # min
            pd.Int64Dtype(),  # max
            pd.Int64Dtype(),  # clusters
            np.float64,       # largest
            np.float64,       # noise
            np.float64,       # time
            ]

        content = []
        for field in Record._fields:
            content.append([
                record.__getattribute__(field)
                for record in self._list
                ])

        return make_typed_DataFrame(
            columns=Record._fields,
            dtypes=_record_dtypes,
            content=content,
            )


def make_typed_DataFrame(columns, dtypes, content=None):
    """Construct :obj:`pandas.DataFrame` with typed columns"""

    if not PANDAS_FOUND:
        raise ModuleNotFoundError("No module named 'pandas'")

    assert len(columns) == len(dtypes)

    if content is None:
        content = [[] for i in range(len(columns))]

    df = pd.DataFrame({
        k: pd.array(c, dtype=v)
        for k, v, c in zip(columns, dtypes, content)
        })

    return df