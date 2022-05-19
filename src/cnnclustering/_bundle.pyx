from collections.abc import Iterable, MutableMapping, MutableSequence
import weakref

try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    NX_FOUND = False

import numpy as np
cimport numpy as np



from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from cnnclustering._types import InputData
from cnnclustering.report import Summary


cdef class Bundle:

    def __cinit__(
            self,
            input_data=None,
            graph=None,
            labels=None,
            reference_indices=None,
            children=None,
            alias=None,
            parent=None,
            meta=None,
            summary=None,
            hierarchy_level=0):

        self.input_data = input_data
        self._graph = graph  # TODO: Not a property yet
        self.labels = labels
        self._reference_indices = reference_indices
        self.children = children

        if alias is None:
            alias = "root"
        self.alias = str(alias)

        if parent is not None:
            self._parent = weakref.proxy(parent)
            self.hierarchy_level = parent.hierarchy_level + 1
        else:
            self.hierarchy_level = hierarchy_level

        self.meta = meta
        self.summary = summary

    @property
    def labels(self):
        if self._labels is not None:
            return self._labels.labels
        return None

    @labels.setter
    def labels(self, value):
        if (value is not None) & (not isinstance(value, Labels)):
            value = Labels(value)
        self._labels = value

    @property
    def root_indices(self):
        if self._reference_indices is not None:
            return self._reference_indices.root
        return None

    @property
    def parent_indices(self):
        if self._reference_indices is not None:
            return self._reference_indices.parent
        return None

    @property
    def input_data(self):
        if self._input_data is not None:
            return self._input_data.data
        return None

    @input_data.setter
    def input_data(self, value):
        if (value is not None) & (not isinstance(value, InputData)):
            raise TypeError(
                f"Can't use object of type {type(value).__name__} as input data. "
                f"Expected type {InputData.__name__}."
                )
        self._input_data = value

    @property
    def children(self):
        """
        Return a mapping of child cluster labels to
        :obj:`cnnclustering.bundle.Bundle` instances representing
        the children of this clustering.
        """
        return self._children

    @children.setter
    def children(self, value):
        """
        Return a mapping of child cluster labels to
        :obj:`cnnclustering.bundle.Bundle` instances representing
        the children of this clustering.
        """
        if value is None:
            value = {}
        if not isinstance(value, MutableMapping):
            raise TypeError("Expected a mutable mapping")
        self._children = value

    @property
    def hierarchy_level(self):
        """
        The level of this clustering in the hierarchical
        tree of clusterings (0 for the root instance).
        """
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, value):
        self._hierarchy_level = int(value)

    @property
    def summary(self):
        """
        Return an instance of :obj:`cnnclustering.cluster.Summary`
        collecting clustering results for this clustering.
        """
        return self._summary

    @summary.setter
    def summary(self, value):
        """
        Return an instance of :obj:`cnnclustering.cluster.Summary`
        collecting clustering results for this clustering.
        """
        if value is None:
            value = Summary()
        if not isinstance(value, MutableSequence):
            raise TypeError("Expected a mutable sequence")
        self._summary = value

    def info(self):
        access = []
        if self._input_data is not None:
            for kind, check in (
                    ("coordinates", "access_coords"),
                    ("distances", "access_distances"),
                    ("neighbours", "access_neighbours")):

                if self._input_data.meta.get(check, False):
                    access.append(kind)

        if not access:
            access = ["unknown"]

        n_points = self._input_data.n_points if self._input_data is not None else None

        attr_str = "\n".join([
            f"alias: {self.alias!r}",
            f"hierarchy_level: {self._hierarchy_level}",
            f"access: {', '.join(access)}",
            f"points: {n_points}",
            f"children: {len(self.children)}",
        ])

        return attr_str

    def get_child(self, label):
        """Retrieve a child of this bundle

        Args:
            label:
                Can be

                    * an integer in which case the child with the respective
                      label is returned
                    * a list of integers in which case the hierarchy of children
                      is traversed and the last child is returned
                    * a string of integers separated by a dot (e.g. "1.1.2") which
                      will be interpreted as a  list of integers (e.g. [1, 1, 2])

        Returns:
            A :obj:`~cnnclustering._bundle.Bundle`

        Note:
            It is not checked if a children mapping exists.
        """
        if isinstance(label, str):
            label = [int(l) for l in label.split(".")]

        if isinstance(label, Iterable) and (len(label) == 1):
            label = label[0]

        if isinstance(label, int):
            try:
                return self._children[label]
            except KeyError:
                raise KeyError(
                    f"Clustering {self.alias!r} has no child with label {label}"
                    )

        next_label, *rest = label
        try:
            return self._children[next_label].get_child(rest)
        except KeyError:
            raise KeyError(
                f"Clustering {self.alias!r} has no child with label {next_label}"
                )

    cpdef void add_child(self, AINDEX label: int):
        """Add a child for this bundle

        Args:
            label: Add child with this label

        Note:
            If the label already exists, the respective child is silently
            overridden. It is not checked if a children mapping exists, either.
        """
        self._children[label] = type(self)(parent=self)

    cpdef void isolate(
            self,
            bint purge: bool = True,
            bint isolate_input_data: bool = True):
        """Create a child for each existing cluster label

        Note:
            see :func:`~cnnclustering._bundle.isolate`
        """

        isolate(self, purge, isolate_input_data)


cpdef void isolate(
        Bundle bundle,
        bint purge: bool = True,
        bint isolate_input_data: bool = True):
    """Create child clusterings from cluster labels

    Args:
        bundle: A bundle to operate on.
        purge: If `True`, creates a new mapping for the children of this
            clustering.
        isolate_input_data: If `True`, attaches a subset of the input data
            of this clustering to the child.
    """

    cdef AINDEX label, index
    cdef list indices
    cdef AINDEX index_part_end, child_index_part_end

    if purge or (bundle._children is None):
        bundle._children = {}

    for label, indices in bundle._labels.mapping.items():
        # Assume indices to be sorted
        parent_indices = np.array(indices, dtype=np.intp)
        if bundle._reference_indices is None:
            root_indices = parent_indices
        else:
            root_indices = bundle._reference_indices.root[parent_indices]

        bundle.add_child(label)
        bundle._children[label]._reference_indices = ReferenceIndices(
            root_indices,
            parent_indices
            )
        parent_alias = bundle.alias if bundle.alias is not None else ""
        bundle._children[label].alias += f"{parent_alias} - {label}"

        if not isolate_input_data:
            continue

        bundle._children[label]._input_data = bundle._input_data.get_subset(
            indices
            )

        edges = bundle._input_data.meta.get("edges", None)
        if edges is None:
            continue

        bundle._children[label]._input_data.meta["edges"] = child_edges = []

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