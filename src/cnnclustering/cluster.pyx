from collections import Counter, defaultdict
from collections.abc import MutableSequence, Iterable
import functools
from operator import itemgetter
import time
from typing import Any, Optional, Type, Union
from typing import Container, List, Tuple, Sequence
import weakref

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from . import plot
    MPL_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    MPL_FOUND = False

try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    NX_FOUND = False

import numpy as np
cimport numpy as np

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
    InputDataExtNeighboursMemoryview,
    NeighboursGetterBruteForce,
    NeighboursGetterLookup,
    NeighboursGetterExtLookup,
    NeighboursGetterExtBruteForce,
    NeighboursList,
    NeighboursSet,
    NeighboursExtVector,
    MetricDummy,
    MetricPrecomputed,
    MetricEuclidean,
    MetricExtDummy,
    MetricExtPrecomputed,
    MetricExtEuclidean,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    QueueFIFODeque,
    QueueExtFIFOQueue,
)
from cnnclustering._fit import FitterExtBFS, FitterBFS, PredictorFirstmatch

from libcpp.vector cimport vector as cppvector


COMPONENT_NAME_TYPE_MAP = {
    "input_data": {
        "points_array2d": InputDataExtPointsMemoryview,
        "neighbourhoods_array2d": InputDataExtNeighboursMemoryview
    },
    "neighbours_getter": {
        "brute_force": NeighboursGetterExtBruteForce,
        "lookup": NeighboursGetterExtLookup,
    },
    "neighbours": {
        "vector": NeighboursExtVector,
    },
    "metric": {
        "dummy": MetricExtDummy,
        "precomputed": MetricExtPrecomputed,
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


registered_recipes = {
    "from_points_brute_force": {
        "input_data": "points_array2d",
        "neighbours_getter": "brute_force",
        "neighbours": ("vector", (10,), {}),
        "neighbour_neighbours": ("vector", (10,), {}),
        "metric": "euclidean",
        "similarity_checker": "contains",
        "queue": "fifo",
        "fitter": "bfs",
    },
    "from_neighbourhoods_lookup": {
        "input_data": "neighbourhoods_array2d",
        "neighbours_getter": "lookup",
        "neighbours": ("vector", (10,), {}),
        "neighbour_neighbours": ("vector", (10,), {}),
        "metric": "dummy",
        "similarity_checker": "contains",
        "queue": "fifo",
        "fitter": "bfs",
    }
}

def prepare_clustering(data, preparation_hook=None, **recipe):
    """Initialise clustering with input data

    Args:
        data: Data that should be clustered in a format
            compatible with 'input_data' specified in the building
            `recipe`.  May go through `preparation_hook` to establish
            compatibility.

    Keyword args:
        preparation_hook: A function that takes input data as a
            single argument and returns the (optionally) reformatted data
            plus additional information (e.g. "meta") in form of
            an argument tuple and a keyword argument dictionary that can
            be used to initialise an input data type.
            If `None` uses :meth:`prepare_points_from_parts`.
        recipe: Building instructions for a
            :obj:`cnnclustering.cluster.Clustering` instance.
    """

    default_recipe = {
        **registered_recipes["from_points_brute_force"]
        }

    default_recipe.update(recipe)

    if preparation_hook is not None:
        data_args, data_kwargs = preparation_hook(data)
    else:
        data_args, data_kwargs = prepare_points_from_parts(data)

    if data_args is None:
        data_args = (data,)

    if data_kwargs is None:
        data_kwargs = {"meta": {}}

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
            args = (*data_args, *args)

            data_kwargs["meta"].update(kwargs.get("meta", {}))
            kwargs.update(data_kwargs)

        if component_type is not None:
            components[component_kw] = component_type(
                *args, **kwargs
            )

    return Clustering(**components)


def prepare_pass(data):
    """Dummy preparation hook

    Use if not preparation of input data is desired.

    Args:
        data: Input data that should be prepared.

    Returns:
        None, None
    """

    return None, None


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


class Clustering:
    """Represents a clustering endeavour

    A clustering object is made by composition of all necessary parts to
    carry out a clustering of input data points.

    Note:
        A clustering instance may also be created using the convenience
        function
        :func:`cnnclustering.cluster.prepare_clustering`

    Args:
        input_data: Any object implementing the input data interface.
            Represents the data points to be clustered.
        neighbours_getter: Any object implementing the neighbours
            getter interface. Controls how neighbours are
            retrieved/calculated from `input data`.
        neighbours: Any object implementing the neighbours
            interface.
            Represents neighbours found by the `neighbours_getter`.
        neighbour_neighbours: Same as `neighbours` but used for the
            neighbours of the neighbours.
        metric: Any object implementing the metric interface. Can be
            used by `neighbours_getter` to retrieved/calculated
            `neighbours` from `input data`.
        similarity_checker: Any object implementing the similarity
            checker interface. Evaluates if to points in `input_data`
            are part of the same cluster based on their `neighbours`.
        queue: Any object implementing the queue interface. May be
            used during the clustering procedure.
        fitter: Any object implementing the fitter interface. Executes
            the clustering procedure.
        predictor: Any object implementing the predictor interface.
            Translates a clustering result to another
            :obj:`cnnclustering.cluster.Clustering` object with different
            `input_data`.
        labels: An instance of :obj:`cnnclustering._types.Labels` holding
            cluster label assignments for points in `input_data`.
        alias: A descriptive string identifier associated with this
            clustering.
        parent: If not None, an instance of :obj:`cnnclustering.cluster.Clustering`
            of which this clustering is a child of.

    Attributes:
        input_data: A representation of the input data, typically a
            (list of) NumPy array(s). Shorthand for
            :obj:`self._input_data.data
        hierarchy_level: The level of this clustering in the hierarchical
            tree of clusterings (0 for the root instance).
        labels: An instance of :obj:`cnnclustering._types.Labels` holding
            cluster label assignments for points in `input_data`.
        children: A dictionary with child cluster labels as keys and
            :obj:`cnnclustering.cluster.Clustering` instances as values.
        summary: An instance of :obj:`cnnclustering.cluster.Summary`
            collecting clustering results.
    """

    take_over_attrs = [
        "_neighbours_getter",
        "_neighbours",
        "_neighbour_neighbours",
        "_metric",
        "_similarity_checker",
        "_queue",
        "_fitter",
        ]

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
            predictor=None,
            labels=None,
            alias: str = "root",
            parent=None):

        self._input_data = input_data
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._metric = metric
        self._similarity_checker = similarity_checker
        self._queue = queue
        self._fitter = fitter
        self._predictor = predictor
        self._labels = labels

        self._children = None
        self._root_indices = None
        self._parent_indices = None

        self._summary = Summary()

        self.alias = alias

        if parent is not None:
            self.parent = weakref.proxy(parent)

            for attr in self.take_over_attrs:
                if getattr(self, attr) is None:
                    setattr(self, attr, getattr(self.parent, attr))

            self.hierarchy_level = parent.hierarchy_level + 1

        else:
            self.hierarchy_level = 0

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if not isinstance(value, Labels):
            value = Labels(value)
        self._labels = value

    @property
    def children(self):
        return self._children

    @property
    def hierarchy_level(self):
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, value):
        self._hierarchy_level = int(value)

    @property
    def input_data(self):
        if self._input_data is not None:
            return self._input_data.data
        return

    @property
    def summary(self):
        return self._summary

    def __repr__(self):
        attr_repr = ", ".join([
            f"input_data={self._input_data!r}",
            f"neighbours_getter={self._neighbours_getter!r}",
            f"neighbours={self._neighbours!r}",
            f"neighbour_neighbours={self._neighbour_neighbours!r}",
            f"metric={self._metric!r}",
            f"similarity_checker={self._similarity_checker!r}",
            f"queue={self._queue!r}",
            f"fitter={self._fitter!r}",
            f"predictor={self._predictor!r}",
        ])

        return f"{type(self).__name__}({attr_repr})"

    def __str__(self):
        try:
            input_data_kind = self._input_data.meta["kind"]
        except KeyError, AttributeError:
            input_data_kind = "unknown"

        n_points = self._input_data.n_points if self._input_data is not None else None
        n_children = len(self._children) if self._children is not None else None

        attr_str = "\n".join([
            f"alias: {self.alias!r}",
            f"hierarchy_level: {self._hierarchy_level}",
            f"input_data_kind: {input_data_kind}",
            f"points: {n_points}",
            f"children: {n_children}",
        ])

        return attr_str

    def get_child(self, label):
        if self._children is None:
            raise LookupError("Clustering has no children")

        if isinstance(label, str):
            label = [int(l) for l in label.split(".")]

        if isinstance(label, Iterable) and (len(label) == 1):
            label = label[0]

        if isinstance(label, int):
            if label in self._children:
                return self._children[label]
            raise KeyError(
                f"Clustering {self.alias!r} has no child with label {label}"
                )

        next_label, *rest = label
        if next_label in self._children:
            return self._children[next_label].get_child(rest)
        raise KeyError(
            f"Clustering {self.alias!r} has no child with label {next_label}"
            )

    def _fit(self, cluster_params: Type["ClusterParameters"]) -> None:
        """Execute clustering procedure

        Low-level alternative to
        :meth:`cnnclustering.cluster.Clustering.fit`.
        """

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

    def fit(
            self,
            radius_cutoff: float,
            cnn_cutoff: int,
            member_cutoff: int = None,
            max_clusters: int = None,
            cnn_offset: int = None,
            sort_by_size: bool = True,
            info: bool = True,
            record: bool = True,
            record_time: bool = True,
            v: bool = True,
            purge: bool = False) -> None:
        """Execute clustering procedure

        Args:
            radius_cutoff: Neighbour search radius.
            cnn_cutoff: Similarity criterion.
            member_cutoff: Valid clusters need to have at least this
                many members.  Passed on  to :meth:`Labels.sort_by_size`
                if `sort_by_size` is `True`.  Has no effect otherwise
                and valid clusters have at least one member.
            max_clusters: Keep only the largest `max_clusters` clusters.
                Passed on to :meth:`Labels.sort_by_size` if
                `sort_by_size` is `True`.  Has no effect otherwise.
            cnn_offset: Exists for compatibility reasons and is
                substracted from `cnn_cutoff`.  If `cnn_offset = 0`, two
                points need to share at least `cnn_cutoff` neighbours
                to be part of the same cluster without counting any of
                the two points.  In former versions of the clustering,
                self-counting was included and `cnn_cutoff = 2` is
                equivalent to `cnn_cutoff = 0` in this version.
            sort_by_size: Weather to sort (and trim) the created
                :obj:`Labels` instance.  See also
                :meth:`Labels.sort_by_size`.
            info: Wether to modify :obj:`Labels.meta` information for
                this clustering.
            record: Wether to create a :obj:`Record`
                instance for this clustering which is appended to the
                :obj:`Summary`.
            record_time: Wether to time clustering execution.
            v: Be chatty.
            purge: If True, force re-initialisation of cluster label
                assignments.
        """

        cdef set old_label_set, new_label_set
        cdef ClusterParameters cluster_params
        cdef AINDEX current_start, _cnn_offset

        if cnn_offset is None:
            _cnn_offset = 0
        else:
            _cnn_offset = cnn_offset

        if (self._labels is None) or purge or (
                not self._labels.meta.get("frozen", False)):

            self._labels = Labels(
                np.zeros(self._input_data.n_points, order="C", dtype=P_AINDEX)
                )
            old_label_set = set()
            current_start = 1
        else:
            old_label_set = self._labels.to_set()
            current_start = max(old_label_set) + 1

        cluster_params = ClusterParameters(
            radius_cutoff,
            cnn_cutoff - _cnn_offset,
            current_start,
            )

        if record_time:
            _, execution_time = timed(self._fitter.fit)(
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
        else:
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
            execution_time = None

        if info:
            new_label_set = self._labels.to_set()
            params = {
                k: (radius_cutoff, cnn_cutoff - _cnn_offset)
                for k in new_label_set - old_label_set
                if k != 0
                }
            meta = {
                "params": params,
                "reference": weakref.proxy(self),
                "origin": "fit"
            }
            old_params = self._labels.meta.get("params", {})
            old_params.update(meta["params"])
            meta["params"] = old_params
            self._labels.meta.update(meta)

        if sort_by_size:
            self._labels.sort_by_size(member_cutoff, max_clusters)

        if record:
            n_noise = 0
            frequencies = Counter(self._labels.labels)

            if 0 in frequencies:
                n_noise = frequencies.pop(0)

            n_largest = frequencies.most_common(1)[0][1] if frequencies else 0

            rec = Record(
                self._input_data.n_points,
                radius_cutoff,
                cnn_cutoff - _cnn_offset,
                member_cutoff,
                max_clusters,
                len(self._labels.to_set() - {0}),
                n_largest / self._input_data.n_points,
                n_noise / self._input_data.n_points,
                execution_time,
                )

            if v:
                print(rec)

            self._summary.append(rec)

        return

    def fit_hierarchical(
            self,
            radius_cutoff: Union[float, List[float]],
            cnn_cutoff: Union[int, List[int]],
            member_cutoff: int = None,
            max_clusters: int = None,
            cnn_offset: int = None):
        """Execute hierarchical clustering procedure

        """

        cdef AINDEX member_count, _member_cutoff

        if member_cutoff is None:
            member_cutoff = 2
        _member_cutoff = member_cutoff

        cdef cppvector[AVALUE] radius_cutoff_vector
        cdef cppvector[AINDEX] cnn_cutoff_vector

        if isinstance(radius_cutoff, float):
            radius_cutoff = [radius_cutoff]

        if isinstance(cnn_cutoff, int):
            cnn_cutoff = [cnn_cutoff]

        if len(radius_cutoff) == 1:
            radius_cutoff *= len(cnn_cutoff)

        if len(cnn_cutoff) == 1:
            cnn_cutoff *= len(radius_cutoff)

        cdef AINDEX step, n_steps = len(radius_cutoff)
        assert n_steps == len(cnn_cutoff)

        radius_cutoff_vector = radius_cutoff
        cnn_cutoff_vector = cnn_cutoff

        cdef ClusterParameters cluster_params
        cdef AINDEX current_start = 1, _cnn_offset

        if cnn_offset is None:
            _cnn_offset = 0
        else:
            _cnn_offset = cnn_offset

        cdef AINDEX index, n = self._input_data.n_points
        cdef AINDEX cluster_label, parent_label
        cdef AINDEX parent_root_ptr, child_root_index, current_max_label

        cdef Labels previous_labels = Labels(
            np.ones(n, order="C", dtype=P_AINDEX)
            )
        cdef Labels current_labels

        self._root_indices = np.arange(n)
        self._parent_indices = np.arange(n)

        cdef dict terminal_cluster_references = {1: self}
        cdef dict new_terminal_cluster_references
        cdef list child_indices, root_indices, parent_indices

        for step in range(n_steps):

            current_labels = Labels(
                np.zeros(n, order="C", dtype=P_AINDEX)
                )

            cluster_params = ClusterParameters(
                radius_cutoff_vector[step],
                cnn_cutoff_vector[step] - _cnn_offset,
                current_start
                )

            self._fitter.fit(
                self._input_data,
                self._neighbours_getter,
                self._neighbours,
                self._neighbour_neighbours,
                self._metric,
                self._similarity_checker,
                self._queue,
                current_labels,
                cluster_params
                )

            current_clusters = defaultdict(lambda: defaultdict(list))
            current_clusters_processed = defaultdict(lambda: defaultdict(list))
            current_max_label = np.max(previous_labels.labels)

            for index in range(n):
                cluster_label = current_labels._labels[index]

                if cluster_label == 0:
                    continue

                current_clusters[previous_labels._labels[index]][cluster_label].append(index)

            for parent_label, child_mapping in current_clusters.items():
                sorted_child_labels_and_membercount = sorted(
                    {
                        k: len(v)
                        for k, v in child_mapping.items()
                    }.items(), key=itemgetter(1), reverse=True
                )

                child_label, member_count = sorted_child_labels_and_membercount[0]
                child_indices = child_mapping[child_label]
                if member_count >= _member_cutoff:
                    current_labels.labels[child_indices] = parent_label  # !
                    current_clusters_processed[parent_label][parent_label] = child_indices
                else:
                    current_labels.labels[child_indices] = 0

                for child_label, member_count in sorted_child_labels_and_membercount[1:]:
                    child_indices = child_mapping[child_label]
                    if member_count >= _member_cutoff:
                        current_labels.labels[child_indices] = child_label + current_max_label
                        current_clusters_processed[parent_label][child_label + current_max_label] = child_indices
                    else:
                        current_labels.labels[child_indices] = 0

            new_terminal_cluster_references = {}
            for parent_label, clustering_instance in terminal_cluster_references.items():
                clustering_instance._labels = Labels.from_sequence(
                    current_labels.labels[clustering_instance._root_indices]
                    )
                clustering_instance._children = defaultdict(
                    lambda: Clustering(parent=clustering_instance)
                    )

                for child_label, root_indices in current_clusters_processed[parent_label].items():
                    clustering_instance._children[child_label]._root_indices = np.asarray(root_indices)
                    clustering_instance._children[child_label]._input_data = self._input_data.get_subset(root_indices)
                    parent_alias = clustering_instance.alias if clustering_instance.alias is not None else ""
                    clustering_instance._children[child_label].alias += f"{parent_alias} - {child_label}"

                    # Reconstruct parent indices
                    parent_root_ptr = 0
                    parent_indices = []
                    for child_root_index in root_indices:
                        while True:
                            if child_root_index == clustering_instance._root_indices[parent_root_ptr]:
                                parent_indices.append(parent_root_ptr)
                                parent_root_ptr += 1
                                break
                            parent_root_ptr += 1

                    clustering_instance._children[child_label]._parent_indices = np.asarray(parent_indices)
                    new_terminal_cluster_references[child_label] = clustering_instance._children[child_label]

            terminal_cluster_references = new_terminal_cluster_references
            previous_labels = current_labels

        return

    def isolate(
            self,
            bint purge: bool = True,
            bint isolate_input_data: bool = True):
        """Create child clusterings from cluster labels

        Args:
            purge: If `True`, creates a new mapping for the children of this
                clustering.
            isolate_input_data: If `True`, attaches a subset of the input data
                of this clustering to the child.
        """

        cdef AINDEX label, index
        cdef list indices

        if purge or (self._children is None):
            self._children = defaultdict(
                lambda: Clustering(parent=self)
                )

        for label, indices in self.labels.mapping.items():
            # Assume indices to be sorted
            parent_indices = np.array(indices, dtype=np.intp)
            if self._root_indices is None:
                root_indices = parent_indices
            else:
                root_indices = self._root_indices[parent_indices]

            self._children[label]._root_indices = root_indices
            self._children[label]._parent_indices = parent_indices
            parent_alias = self.alias if self.alias is not None else ""
            self._children[label].alias += f"{parent_alias} - {label}"

            if not isolate_input_data:
                continue

            self._children[label]._input_data = self._input_data.get_subset(
                indices
                )

            edges = self._input_data.meta.get("edges", None)
            if edges is None:
               continue

            self._children[label]._input_data.meta["edges"] = child_edges = []

            if not edges:
                continue

            edges_iter = iter(edges)
            index_part_end = next(edges_iter)
            child_index_part_end = 0

            for index in range(parent_indices.shape[0]):
                if parent_indices[index] < index_part_end:
                    child_index_part_end += 1
                    continue

                while parent_indices[index] >= index_part_end:
                    child_edges.append(child_index_part_end)
                    index_part_end += next(edges_iter)
                    child_index_part_end = 0

                child_index_part_end += 1

            child_edges.append(child_index_part_end)

            while len(child_edges) < len(edges):
               child_edges.append(0)

        return

    def trim_shrinking_leafs(self):

        def _trim_shrinking(clustering, new=True):

            if clustering._children is None:
                splits = will_split = False
            else:
                label_set = clustering._labels.set
                label_set.discard(0)

                if len(label_set) <= 1:
                    splits = False
                else:
                    splits = True

                will_split = []
                for child in clustering._children.values():
                    will_split.append(
                        _trim_shrinking(child, new=splits)
                        )

                will_split = any(will_split)

            keep = new or will_split or splits
            if not keep:
                clustering._labels = None
                clustering._children = None

            return keep

        _trim_shrinking(self)

        return

    def trim_trivial_leafs(self):
        """Scan cluster hierarchy for removable nodes

        If the cluster label assignments on a clustering are all zero
        (noise), the clustering is considered trivial.  In this case,
        the labels and children are reset to `None`.
        """

        def _trim_trivial(clustering):
            if clustering._labels is None:
                return

            if clustering._labels.set == {0}:
                clustering._labels = None
                clustering._children = None
                return

            if clustering._children is None:
                return

            for child in clustering._children.values():
                _trim_trivial(child)

        _trim_trivial(self)

        return

    def reel(self, depth: Optional[int] = None) -> None:
        """Wrap up label assignments of lower hierarchy levels

        Args:
            depth: How many lower levels to consider. If `None`,
            consider all.
        """

        cdef AINDEX label, new_label, parent_index

        def _reel(parent, depth):
            if parent._children is None:
                return

            if depth is not None:
                depth -= 1

            parent._labels.meta["origin"] = "reel"
            parent_labels = parent._labels.labels

            for label, child in parent._children.items():
                if child._labels is None:
                    continue

                if (depth is None) or (depth > 0):
                    _reel(child, depth)

                n_clusters = max(parent_labels)

                child_labels = child._labels.labels
                for index, old_label in enumerate(child_labels):
                    if old_label == 0:
                        new_label = 0
                    else:
                        new_label = old_label + n_clusters

                    parent_index = child._parent_indices[index]
                    parent_labels[parent_index] = new_label

                try:
                    _ = parent._labels.meta["params"].pop(label)
                except KeyError:
                    pass

                params = child._labels.meta.get("params", {})
                for old_label, p in params.items():
                    parent._labels.meta["params"][old_label + n_clusters] = p

        if depth is not None:
            assert depth > 0

        _reel(self, depth)

        return

    def predict(
            self,
            other: Type["Clustering"],
            radius_cutoff: float,
            cnn_cutoff: int,
            clusters: Optional[Sequence[int]] = None,
            cnn_offset: Optional[int] = None,
            info: bool = True,
            record: bool = True,
            record_time: bool = True,
            v: bool = True,
            purge: bool = False):
        """Execute prediction procedure

        Args:
            other: :obj:`cnnclustering.cluster.Clustering` instance for
                which cluster labels should be predicted.
            radius_cutoff: Neighbour search radius.
            cnn_cutoff: Similarity criterion.
            cluster: Sequence of cluster labels that should be included
               in the prediction.
            cnn_offset: Exists for compatibility reasons and is
                substracted from `cnn_cutoff`.  If `cnn_offset = 0`, two
                points need to share at least `cnn_cutoff` neighbours
                to be part of the same cluster without counting any of
                the two points.  In former versions of the clustering,
                self-counting was included and `cnn_cutoff = 2` is
                equivalent to `cnn_cutoff = 0` in this version.
            purge: If True, force re-initialisation of predicted cluster
                labels.
        """

        cdef ClusterParameters cluster_params

        if cnn_offset is None:
            _cnn_offset = 0
        else:
            _cnn_offset = cnn_offset

        if (other._labels is None) or purge or (
                not other._labels.meta.get("frozen", False)):
            other._labels = Labels(
                np.zeros(other._input_data.n_points, order="C", dtype=P_AINDEX)
                )

        cluster_params = ClusterParameters(
            radius_cutoff,
            cnn_cutoff - _cnn_offset,
            0,
            )

        if clusters is None:
           clusters = self._labels.to_set() - {0}

        other._labels.consider_set = clusters

        if record_time:
            _, execution_time = timed(self._predictor.predict)(
                self._input_data,
                other._input_data,
                self._neighbours_getter,
                other._neighbours_getter,
                other._neighbours,
                other._neighbour_neighbours,
                other._metric,
                other._similarity_checker,
                self._labels,
                other._labels,
                cluster_params
                )
        else:
            self._predictor.predict(
                self._input_data,
                other._input_data,
                self._neighbours_getter,
                other._neighbours_getter,
                other._neighbours,
                other._neighbour_neighbours,
                other._metric,
                other._similarity_checker,
                self._labels,
                other._labels,
                cluster_params
                )
            execution_time = None

        if info:
            params = {
                k: (radius_cutoff, cnn_cutoff - _cnn_offset)
                for k in clusters
                if k != 0
                }
            meta = {
                "params": params,
                "reference": weakref.proxy(self),
                "origin": "predict",
            }
            old_params = other._labels.meta.get("params", {})
            old_params.update(meta["params"])
            meta["params"] = old_params
            other._labels.meta.update(meta)

        other._labels.meta["frozen"] = True

    def summarize(
            self,
            ax=None,
            quantity: str = "execution_time",
            treat_nan: Optional[Any] = None,
            convert: Optional[Any] = None,
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

        if (self._summary is None) or (len(self._summary._list) == 0):
            raise LookupError(
                "No records in summary"
                )

        ax_props_defaults = {
            "xlabel": "$r$",
            "ylabel": "$c$",
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
            ax, self._summary.to_DataFrame(),
            quantity=quantity,
            treat_nan=treat_nan,
            convert=convert,
            contour_props=contour_props_defaults
            )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted

    def pie(self, ax=None, pie_props=None):

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plot.pie(self, ax=ax, pie_props=pie_props)

        return fig, ax

    def tree(self, ax=None, ignore=None, pos_props=None, draw_props=None):

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        graph = self.to_nx_DiGraph(ignore=ignore)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        pos_props_defaults = {
            "source": "1",
        }

        if pos_props is not None:
            pos_props_defaults.update(pos_props)

        shortened_labels = {}
        for key in graph.nodes.keys():
            skey = key.rsplit(".", 1)
            shortened_labels[key] = skey[len(skey) - 1]

        draw_props_defaults = {
            "labels": shortened_labels,
            "with_labels": True,
            "node_shape": "s",
            "edgecolors": "k",
        }

        if draw_props is not None:
            draw_props_defaults.update(draw_props)

        plot.plot_graph_sugiyama_straight(
            graph, ax=ax,
            pos_props=pos_props_defaults,
            draw_props=draw_props_defaults,
            )

        return fig, ax

    def evaluate(
            self,
            ax=None,
            clusters: Optional[Container[int]] = None,
            original: bool = False,
            plot_style: str = 'dots',
            parts: Optional[Tuple[Optional[int]]] = None,
            points: Optional[Tuple[Optional[int]]] = None,
            dim: Optional[Tuple[int, int]] = None,
            mask: Optional[Sequence[Union[bool, int]]] = None,
            ax_props: Optional[dict] = None,
            annotate: bool = True,
            annotate_pos: str = "mean",
            annotate_props: Optional[dict] = None,
            plot_props: Optional[dict] = None,
            plot_noise_props: Optional[dict] = None,
            hist_props: Optional[dict] = None,
            free_energy: bool = True):

        """Returns a 2D plot of an original data set or a cluster result

        Args: ax: The `Axes` instance to which to add the plot.  If
            `None`, a new `Figure` with `Axes` will be created.

            clusters:
                Cluster numbers to include in the plot.  If `None`,
                consider all.

            original:
                Allows to plot the original data instead of a cluster
                result.  Overrides `clusters`.  Will be considered
                `True`, if no cluster result is present.

            plot_style:
                The kind of plotting method to use.

                    * "dots", :func:`ax.plot`
                    * "scatter", :func:`ax.scatter`
                    * "contour", :func:`ax.contour`
                    * "contourf", :func:`ax.contourf`

            parts:
                Use a slice (start, stop, stride) on the data parts
                before plotting. Will be applied before a slice on `points`.

            points:
                Use a slice (start, stop, stride) on the data points
                before plotting.

            dim:
                Use these two dimensions for plotting.  If `None`, uses
                (0, 1).

            mask:
                Sequence of boolean or integer values used for optional
                fancy indexing on the point data array.  Note, that this
                is applied after regular slicing (e.g. via `points`) and
                requires a copy of the indexed data (may be slow and
                memory intensive for big data sets).

            annotate:
                If there is a cluster result, plot the cluster numbers.
                Uses `annotate_pos` to determinte the position of the
                annotations.

            annotate_pos:
                Where to put the cluster number annotation.
                Can be one of:

                    * "mean", Use the cluster mean
                    * "random", Use a random point of the cluster

                Alternatively a list of x, y positions can be passed to
                set a specific point for each cluster
                (*Not yet implemented*)

            annotate_props:
                Dictionary of keyword arguments passed to
                :func:`ax.annotate`.

            ax_props:
                Dictionary of `ax` properties to apply after
                plotting via :func:`ax.set(**ax_props)`.  If `None`,
                uses defaults that can be also defined in
                the configuration file (*Note yet implemented*).

            plot_props:
                Dictionary of keyword arguments passed to various
                functions (:func:`plot.plot_dots` etc.) with different
                meaning to format cluster plotting.  If `None`, uses
                defaults that can be also defined in
                the configuration file (*Note yet implemented*).

            plot_noise_props:
                Like `plot_props` but for formatting noise point
                plotting.

            hist_props:
               Dictionary of keyword arguments passed to functions that
               involve the computing of a histogram via
               `numpy.histogram2d`.

            free_energy:
                If `True`, converts computed histograms to pseudo free
                energy surfaces.

        Returns:
            Figure, Axes and a list of plotted elements
        """

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if (self._input_data is None) or (
                self._input_data.meta.get("kind", None) != "points"):
            raise ValueError(
                "No data points found to evaluate."
            )

        if dim is None:
            dim = (0, 1)
        elif dim[1] < dim[0]:
            dim = dim[::-1]

        if parts is not None:
            by_parts = list(self._input_data.by_parts())[slice(*parts)]
            data = np.vstack(by_parts)
        else:
            data = self._input_data.data

        if points is None:
            points = (None, None, None)

        # Slicing without copying
        data = data[
            slice(*points),
            slice(dim[0], dim[1] + 1, dim[1] - dim[0])
            ]

        if mask is not None:
            data = data[mask]

        # Plot original set or points per cluster?
        cluster_map = None
        if not original:
            if self._labels is not None:
                cluster_map = self._labels.mapping
                if clusters is None:
                    clusters = list(cluster_map.keys())
            else:
                original = True

        ax_props_defaults = {
            "xlabel": "$x$",
            "ylabel": "$y$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        annotate_props_defaults = {}

        if annotate_props is not None:
            annotate_props_defaults.update(annotate_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if plot_style == "dots":
            plot_props_defaults = {
                'lw': 0,
                'marker': '.',
                'markersize': 5,
                'markeredgecolor': 'none',
                }

            if plot_props is not None:
                plot_props_defaults.update(plot_props)

            plot_noise_props_defaults = {
                'color': 'none',
                'lw': 0,
                'marker': '.',
                'markersize': 4,
                'markerfacecolor': 'k',
                'markeredgecolor': 'none',
                'alpha': 0.3
                }

            if plot_noise_props is not None:
                plot_noise_props_defaults.update(plot_noise_props)

            plotted = plot.plot_dots(
                ax=ax, data=data, original=original,
                cluster_map=cluster_map,
                clusters=clusters,
                dot_props=plot_props_defaults,
                dot_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        elif plot_style == "scatter":
            plot_props_defaults = {
                's': 10,
            }

            if plot_props is not None:
                plot_props_defaults.update(plot_props)

            plot_noise_props_defaults = {
                'color': 'k',
                's': 10,
                'alpha': 0.5
            }

            if plot_noise_props is not None:
                plot_noise_props_defaults.update(plot_noise_props)

            plotted = plot.plot_scatter(
                ax=ax, data=data, original=original,
                cluster_map=cluster_map,
                clusters=clusters,
                scatter_props=plot_props_defaults,
                scatter_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        if plot_style in ["contour", "contourf", "histogram"]:

            hist_props_defaults = {
                "avoid_zero_count": False,
                "mass": True,
                "mids": True
            }

            if hist_props is not None:
                hist_props_defaults.update(hist_props)

            if plot_style == "contour":

                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_contour(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot_style == "contourf":
                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_contourf(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot_style == "histogram":
                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_histogram2d(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

        ax.set(**ax_props_defaults)

        return fig, ax

    def to_nx_DiGraph(self, ignore=None):
        """Convert cluster hierarchy to networkx DiGraph

        Keyword args:
            ignore: A set of label not to include into the graph.  Use
                for example to exclude noise (label 0).
        """

        if not NX_FOUND:
            raise ModuleNotFoundError("No module named 'networkx'")

        def add_children(clustering_label, clustering, graph):
            for child_label, child_clustering in sorted(clustering._children.items()):

                if child_label in ignore:
                    continue

                padded_child_label = ".".join([clustering_label, str(child_label)])
                graph.add_node(padded_child_label, object=child_clustering)
                graph.add_edge(clustering_label, padded_child_label)

                if child_clustering._children is not None:
                    add_children(padded_child_label, child_clustering, graph)

        if ignore is None:
            ignore = set()

        if not isinstance(ignore, set):
            ignore = set(ignore)

        graph = nx.DiGraph()
        graph.add_node("1", object=self)
        add_children("1", self, graph)

        return graph


class Record:
    """Cluster result container

    :obj:`cnnclustering.cluster.Record` instances can created during
    :meth:`cnnclustering.cluster.Clustering.fit` and
    are collected in :obj:`cnnclustering.cluster.Summary`.
    """

    __slots__ = [
        "n_points",
        "radius_cutoff",
        "cnn_cutoff",
        "member_cutoff",
        "max_clusters",
        "n_clusters",
        "ratio_largest",
        "ratio_noise",
        "execution_time",
        ]

    def __init__(
            self,
            n_points=None,
            radius_cutoff=None,
            cnn_cutoff=None,
            member_cutoff=None,
            max_clusters=None,
            n_clusters=None,
            ratio_largest=None,
            ratio_noise=None,
            execution_time=None):

        self.n_points = n_points
        self.radius_cutoff = radius_cutoff
        self.cnn_cutoff = cnn_cutoff
        self.member_cutoff = member_cutoff
        self.max_clusters = max_clusters
        self.n_clusters = n_clusters
        self.ratio_largest = ratio_largest
        self.ratio_noise = ratio_noise
        self.execution_time = execution_time

    def __repr__(self):
        attrs_str = ", ".join(
            [
                f"{attr}={getattr(self, attr)!r}"
                for attr in self.__slots__
            ]
            )
        return f"{type(self).__name__}({attrs_str})"

    def __str__(self):
        attr_str = ""
        for attr in self.__slots__:
            if attr == "execution_time":
               continue

            value = getattr(self, attr)
            if value is None:
                attr_str += f"{value!r:<10}"
            elif isinstance(value, float):
                attr_str += f"{value:<10.3f}"
            else:
                attr_str += f"{value:<10}"

        if self.execution_time is not None:
            hours, rest = divmod(self.execution_time, 3600)
            minutes, seconds = divmod(rest, 60)
            execution_time_str = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:.3f}"
        else:
            execution_time_str = None

        printable = (
            f'{"-" * 95}\n'
            f"#points   "
            f"r         "
            f"c         "
            f"min       "
            f"max       "
            f"#clusters "
            f"%largest  "
            f"%noise    "
            f"time     \n"
            f"{attr_str}"
            f"{execution_time_str}\n"
            f'{"-" * 95}\n'
        )
        return printable

    def to_dict(self):
        return {
            "n_points": self.n_points,
            "radius_cutoff":  self.radius_cutoff,
            "cnn_cutoff":  self.cnn_cutoff,
            "member_cutoff":  self.member_cutoff,
            "max_clusters":  self.max_clusters,
            "n_clusters":  self.n_clusters,
            "ratio_largest":  self.ratio_largest,
            "ratio_noise":  self.ratio_noise,
            "execution_time":  self.execution_time,
            }


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
        for field in Record.__slots__:
            content.append([
                record.__getattribute__(field)
                for record in self._list
                ])

        return make_typed_DataFrame(
            columns=Record.__slots__,
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


def timed(function):
    """Decorator to measure execution time"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        go = time.time()
        wrapped_return = function(*args, **kwargs)
        stop = time.time()

        return wrapped_return, stop - go
    return wrapper
