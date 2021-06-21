from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from itertools import count
from typing import Any, Optional, Type
from typing import Container, Iterator, Sequence

import numpy as np

try:
    import sklearn.neighbors
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from libc.math cimport sqrt as csqrt, pow as cpow, fabs as cfabs
from cython.operator cimport dereference, preincrement

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value) nogil


cdef class ClusterParameters:
    """Input parameters for clustering procedure

    Args:
        radius_cutoff: Neighbour search radius :math:`r`.
        cnn_cutoff: Common-nearest-neighbours requirement :math:`c`
            (similarity criterion).

    Keyword args:
        current_start: Use this as the first label for identified clusters.
    """

    def __cinit__(self, radius_cutoff, cnn_cutoff, current_start=1):
        self.radius_cutoff = radius_cutoff
        self.cnn_cutoff = cnn_cutoff
        self.current_start = current_start

    def to_dict(self):
        return {
            "radius_cutoff": self.radius_cutoff,
            "cnn_cutoff": self.cnn_cutoff,
            "current_start": self.current_start
            }

    def __repr__(self):
        return f"{self.to_dict()!r}"

    def __str__(self):
        return f"{self.to_dict()!s}"


cdef class Labels:
    """Represents cluster label assignments

    Args:
        labels: A container of integer cluster labels
            supporting the buffer protocol

    Keyword args:
        consider: A boolean (uint8) container of same length as `labels`
            indicating if a cluster label shoud be considered for assignment
            during clustering.  If `None`, will be created as all true.
        meta: Meta information.  If `None`, will be created as empty
            dictionary.

    Attributes:
        n_points: The length of the labels container
        meta: The meta information dictionary
        labels: The labels container converted to a NumPy ndarray
        consider: The consider container converted to a NumPy ndarray
        mapping: A mapping of cluster labels to indices in `labels`
        set: The set of cluster labels
        consider_set: A set of cluster labels to consider for cluster
            label assignments

    Methods:
        from_sequence: Alternative to construct from a labels sequence
            (not supporting the buffer protocol)
        to_mapping: Convert labels to `mapping`
        to_set: Convert labels to `set`
        sort_by_size: Sort cluster labels by member count (1 being
            the largest cluster)

    """

    def __cinit__(self, labels, *, consider=None, meta=None):

        self._labels = labels
        if consider is None:
            self._consider = np.ones_like(self._labels, dtype=P_ABOOL)
        else:
            self._consider = consider

            if self._labels.shape[0] != self._consider.shape[0]:
                raise ValueError(
                    "'labels' and 'consider' must have the same length"
                    )

        self.n_points = len(labels)

        if meta is None:
            meta = {}
        self.meta = meta

    @property
    def mapping(self):
        return self.to_mapping()

    @property
    def set(self):
        return self.to_set()

    @property
    def labels(self):
        return np.asarray(self._labels)

    @property
    def consider(self):
        return np.asarray(self._consider)

    @property
    def consider_set(self):
        return self._consider_set

    @consider_set.setter
    def consider_set(self, value):
        self._consider_set = value

    def __repr__(self):
        return f"{type(self).__name__}({list(self.labels)!s})"

    def __str__(self):
        return f"{self.labels!s}"

    @classmethod
    def from_sequence(cls, labels, *, consider=None, meta=None):
        labels = np.array(labels, order="C", dtype=P_AINDEX)

        if consider is not None:
            consider = np.array(consider, order="C", dtype=P_ABOOL)

        return cls(labels, consider=consider, meta=meta)

    def to_mapping(self):
        cdef AINDEX index, label

        mapping = defaultdict(list)

        for index in range(self._labels.shape[0]):
            label = self._labels[index]
            mapping[label].append(index)

        return mapping

    def to_set(self):
        cdef AINDEX index, label
        cdef set label_set = set()

        for index in range(self._labels.shape[0]):
            label = self._labels[index]
            label_set.add(label)

        return label_set

    def sort_by_size(
            self,
            member_cutoff: Optional[int] = None,
            max_clusters: Optional[int] = None):
        """Sort labels by clustersize in-place

        Re-assigns cluster numbers so that the biggest cluster (that is
        not noise) is cluster 1.  Also filters clusters out, that have
        not at least `member_cutoff` members.  Optionally, does only
        keep the `max_clusters` largest clusters.

        Args:
           member_cutoff: Valid clusters need to have at least this
              many members.
           max_clusters: Only keep this many clusters.
        """

        cdef AINDEX _max_clusters, _member_cutoff, cluster_count
        cdef AINDEX index, old_label, new_label, member_count
        cdef dict reassign_map, params

        if member_cutoff is None:
            member_cutoff = 2
        _member_cutoff = member_cutoff

        frequencies = Counter(self._labels)
        if 0 in frequencies:
            _ = frequencies.pop(0)

        if frequencies:
            if max_clusters is None:
               _max_clusters = len(frequencies)
            else:
               _max_clusters = max_clusters

            order = frequencies.most_common()
            reassign_map = {}
            reassign_map[0] = 0

            new_labels = count(1)
            for cluster_count, (old_label, member_count) in enumerate(order, 1):
                if cluster_count > _max_clusters:
                    reassign_map[old_label] = 0
                    continue

                if member_count >= _member_cutoff:
                    new_label = next(new_labels)
                    reassign_map[old_label] = new_label
                    continue

                reassign_map[old_label] = 0

            for index in range(self._labels.shape[0]):
                old_label = self._labels[index]
                self._labels[index] = reassign_map[old_label]

            params = self.meta.get("params", {})
            self.meta["params"] = {
                reassign_map[k]: v
                for k, v in params.items()
                if (k in reassign_map) and (reassign_map[k] != 0)
                }

        return

class InputData(ABC):
    """Defines the input data interface"""

    @property
    @abstractmethod
    def data(self):
        """Return underlying data"""

    @property
    @abstractmethod
    def meta(self):
        """Return meta-information"""

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Return total number of points"""

    @property
    @abstractmethod
    def n_dim(self) -> int:
        """Return total number of dimensions"""

    @abstractmethod
    def get_component(self, point: int, dimension: int) -> float:
        """Return one component of point coordinates"""

    @abstractmethod
    def get_n_neighbours(self, point: int) -> int:
        """Return number of neighbours for point"""

    @abstractmethod
    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""

    @abstractmethod
    def get_subset(self, indices: Container) -> Type['InputData']:
        """Return input data subset"""


class InputDataNeighboursSequence(InputData):
    """Implements the input data interface

    Neighbours of points stored as a sequence.
    """

    def __init__(self, data: Sequence, *, meta=None):
        self._data = data
        self._n_points = len(data)
        self._n_dim = 0
        self._n_neighbours = [len(s) for s in self._data]

        _meta = {"access_neighbours": True}
        if meta is not None:
            _meta.update(meta)
        self._meta = _meta

    @property
    def meta(self):
        return self._meta

    @property
    def n_points(self):
        return len(self._data)

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def data(self):
        return self._to_array()

    @property
    def n_neighbours(self):
        return np.asarray(self._n_neighbours)

    def _to_array(self):
        return [np.asarray(list(s)) for s in self._data]

    def get_component(self, point: int, dimension: int) -> float:
        """Not applicable

        Method only present for consistency.
        Returns no relevant information.
        """
        return 0

    def get_n_neighbours(self, point: int) -> int:
        return self._n_neighbours[point]

    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""
        return self._data[point][member]

    def get_subset(self, indices: Container) -> Type['InputDataNeighboursSequence']:
        """Return input data subset"""
        data_subset = [
            [m for m in s if m in indices]
            for i, s in enumerate(self._data)
            if i in indices
        ]

        return type(self)(data_subset)


class InputDataPointsSklearnKDTree(InputData):
    """Implements the input data interface

    Points stored as a NumPy array.  Neighbour queries delicated
    to pre-build KDTree.
    """

    def __init__(self, data: Type[np.ndarray], *, meta=None, **kwargs):
        self._data = data
        self._n_points = self._data.shape[0]
        self._n_dim = self._data.shape[1]

        self.build_tree(**kwargs)
        self.clear_cached()

        _meta = {
            "access_points": True,
            "access_neighbours": True
            }
        if meta is not None:
            _meta.update(meta)
        self._meta = _meta

    @property
    def meta(self):
        return self._meta

    @property
    def n_points(self):
        return self._n_points

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def data(self):
        return self._to_array()

    @property
    def n_neighbours(self):
        return np.asarray(self._n_neighbours)

    def _to_array(self):
        return self._data

    def get_component(self, point: int, dimension: int) -> float:
        return self._data[point, dimension]

    def get_n_neighbours(self, point: int) -> int:
        return self._n_neighbours[point]

    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""
        return self._cached_neighbourhoods[point][member]

    def get_subset(self, indices: Container) -> Type['InputDataNeighboursSequence']:
        """Return input data subset"""
        data_subset = [
            [m for m in s if m in indices]
            for i, s in enumerate(self._data)
            if i in indices
        ]

        return type(self)(data_subset)

    def build_tree(self, **kwargs):
        self._tree = sklearn.neighbors.KDTree(self._data, **kwargs)

    def compute_neighbourhoods(
            self,
            input_data: Type["InputData"],
            radius: float,
            is_sorted: bool = False,
            is_selfcounting: bool = False):

        self._cached_neighbourhoods = self._tree.query_radius(
            input_data._to_array(), r=radius,
            return_distance=False,
            )
        self._radius = radius

        if is_sorted:
            for n in self._cached_neighbourhoods:
                n.sort()

        if is_selfcounting:
            pass

        self._n_neighbours = [len(s) for s in self._cached_neighbourhoods]

    def clear_cached(self):
        self._cached_neighbourhoods = None
        self._n_neighbours = None
        self._radius = None

cdef class InputDataExtNeighboursMemoryview:
    """Implements the input data interface

    Neighbours of points stored as using a memoryview.
    """

    def __cinit__(
            self,
            AINDEX[:, ::1] data not None,
            AINDEX[::1] n_neighbours not None, *, meta=None):

        self._data = data
        self.n_points = self._data.shape[0]
        self.n_dim = 0
        self._n_neighbours = n_neighbours

        _meta = {"access_neighbours": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    @property
    def data(self):
        return self._to_array()

    @property
    def n_neighbours(self):
        return np.asarray(self._n_neighbours)

    def _to_array(self):
        cdef AINDEX i

        return [
            s[:self._n_neighbours[i]]
            for i, s in enumerate(np.asarray(self._data))
            ]

    cdef inline AVALUE _get_component(
            self, AINDEX point, AINDEX dimension) nogil:
        """Not applicable

        Method only present for consistency.
        Returns no relevant information.
        """
        return 0

    def get_component(
            self, point: int, dimension: int) -> float:
        return self._get_component(point, dimension)

    cdef inline AINDEX _get_n_neighbours(self, AINDEX point) nogil:
        return self._n_neighbours[point]

    def get_n_neighbours(self, point: int) -> int:
        return self._get_n_neighbours(point)

    cdef inline AINDEX _get_neighbour(self, AINDEX point, AINDEX member) nogil:
        """Return a member for point"""
        return self._data[point, member]

    def get_neighbour(self, point: int, member: int) -> int:
        return self._get_neighbour(point, member)

    def get_subset(self, indices: Sequence) -> Type['InputDataExtNeighboursMemoryview']:
        """Return input data subset"""

        cdef list lengths

        data_subset = np.asarray(self._data)[indices]
        data_subset = [
            [m for m in a if m in indices]
            for a in data_subset
        ]

        lengths = [len(a) for a in data_subset]
        pad_to = max(lengths)

        for i, a in enumerate(data_subset):
            missing_elements = pad_to - lengths[i]
            a.extend([0] * missing_elements)

        return type(self)(np.asarray(data_subset, order="C", dtype=P_AINDEX))


cdef class InputDataExtPointsMemoryview:
    """Implements the input data interface"""

    def __cinit__(self, AVALUE[:, ::1] data not None, *, meta=None):
        self._data = data
        self.n_points = self._data.shape[0]
        self.n_dim = self._data.shape[1]

        _meta = {"access_points": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    @property
    def data(self):
        return self._to_array()

    def _to_array(self):
        return np.asarray(self._data)

    cdef inline AVALUE _get_component(
            self, AINDEX point, AINDEX dimension) nogil:
        return self._data[point, dimension]

    def get_component(
            self, point: int, dimension: int) -> float:
        return self._get_component(point, dimension)

    cdef inline AINDEX _get_n_neighbours(self, AINDEX point) nogil:
        return 0

    def get_n_neighbours(self, point: int):
        return self._get_n_neighbours(point)

    cdef inline AINDEX _get_neighbour(self, AINDEX point, AINDEX member) nogil:
        return 0

    def get_neighbour(self, point: int, member: int):
        return self._get_neighbour(point, member)

    def get_subset(self, object indices: Sequence) -> Type['InputDataExtPointsMemoryview']:
        """Return input data subset"""
        return type(self)(self.data[indices])
        # Slow because it goes via numpy array

    def by_parts(self) -> Iterator:
        """Yield data by parts

        Returns:
            Generator of 2D :obj:`numpy.ndarray` s (parts)
        """

        if self.n_points > 0:
            edges = self.meta.get("edges", None)
            if not edges:
               edges = [self.n_points]

            data = self.data

            start = 0
            for end in edges:
                yield data[start:(start + end), :]
                start += end

        else:
            yield from ()


InputData.register(InputDataExtPointsMemoryview)


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def assign(self, member: int) -> None:
       """Add a member to this container"""

    @abstractmethod
    def reset(self) -> None:
       """Reset/empty this container"""

    @abstractmethod
    def enough(self, member_cutoff: int) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def contains(self, member: int) -> bool:
       """Return True if member is in neighbours container"""


class NeighboursList(Neighbours):
    def __init__(self, neighbours=None):
        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
        else:
            self.reset()

    @property
    def n_points(self):
        return self._n_points

    def assign(self, member: int):
        self._neighbours.append(member)
        self._n_points += 1

    def reset(self):
        self._neighbours = []
        self._n_points = 0

    def enough(self, member_cutoff: int) -> bool:
        if self._n_points > member_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        return self._neighbours[index]

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


class NeighboursSet(Neighbours):
    def __init__(self, neighbours=None):
        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
            self._query = 0
            self._iter = None
        else:
            self.reset()

    @property
    def n_points(self):
        return self._n_points

    def assign(self, member: int):
        self._neighbours.add(member)
        self._n_points += 1

    def reset(self):
        self._neighbours = set()
        self._n_points = 0
        self._query = 0
        self._iter = None

    def enough(self, member_cutoff: int):
        if self._n_points > member_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        if (self._iter is None) or (index != self._query):
            self._iter = iter(self._neighbours)
            self._query = 0

        while self._query != index:
            _ = next(self._iter)
            self._query += 1

        return next(self._iter)

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


cdef class NeighboursExtVector:
    """Implements the neighbours interface

    Uses an underlying C++ std:vector.

    Args:
        initial_size: Number of elements reserved for the size of vector.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a vector.
    """

    def __cinit__(self, AINDEX initial_size=1, neighbours=None):
        self._initial_size = initial_size

        if neighbours is not None:
            self._neighbours = neighbours
            self.n_points = self._neighbours.size()
            self._neighbours.reserve(self._initial_size)
        else:
            self._reset()

    cdef inline void _assign(self, AINDEX member) nogil:
        self._neighbours.push_back(member)
        self.n_points += 1

    def assign(self, member: int):
        self._assign(member)

    cdef inline void _reset(self) nogil:
        self._neighbours.resize(0)
        self._neighbours.reserve(self._initial_size)
        self.n_points = 0

    def reset(self):
        self._reset()

    cdef inline bint _enough(self, AINDEX member_cutoff) nogil:
        if self.n_points > member_cutoff:
            return True
        return False

    def enough(self, member_cutoff: int):
        return self._enough(member_cutoff)

    cdef inline AINDEX _get_member(self, AINDEX index) nogil:
        return self._neighbours[index]

    def get_member(self, index: int):
        return self._get_member(index)

    cdef inline bint _contains(self, AINDEX member) nogil:

        if find(self._neighbours.begin(), self._neighbours.end(), member) == self._neighbours.end():
            return False
        return True

    def contains(self, member: int):
        return self._contains(member)


cdef class NeighboursExtCPPSet:
    """Implements the neighbours interface

    Uses an underlying C++ std:set.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a C++ set.
    """

    def __cinit__(self, neighbours=None):

        if neighbours is not None:
            self._neighbours = neighbours
            self.n_points = self._neighbours.size()
        else:
            self._reset()

    cdef inline void _assign(self, AINDEX member) nogil:
        self._neighbours.insert(member)
        self.n_points += 1

    def assign(self, member: int):
        self._assign(member)

    cdef inline void _reset(self) nogil:
        self._neighbours.clear()
        self.n_points = 0

    def reset(self):
        self._reset()

    cdef inline bint _enough(self, AINDEX member_cutoff) nogil:
        if self.n_points > member_cutoff:
            return True
        return False

    def enough(self, member_cutoff: int):
        return self._enough(member_cutoff)

    cdef inline AINDEX _get_member(self, AINDEX index) nogil:
        cdef cppset[AINDEX].iterator it = self._neighbours.begin()
        cdef AINDEX i

        for i in range(index):
            preincrement(it)

        return dereference(it)

    def get_member(self, index: int):
        return self._get_member(index)

    cdef inline bint _contains(self, AINDEX member) nogil:
        if self._neighbours.find(member) == self._neighbours.end():
            return False
        return True

    def contains(self, member: int):
        return self._contains(member)


cdef class NeighboursExtCPPUnorderedSet:
    """Implements the neighbours interface

    Uses an underlying C++ std:unordered_set.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a C++ set.
    """

    def __cinit__(self, neighbours=None):

        if neighbours is not None:
            self._neighbours = neighbours
            self.n_points = self._neighbours.size()
        else:
            self._reset()

    cdef inline void _assign(self, AINDEX member) nogil:
        self._neighbours.insert(member)
        self.n_points += 1

    def assign(self, member: int):
        self._assign(member)

    cdef inline void _reset(self) nogil:
        self._neighbours.clear()
        self.n_points = 0

    def reset(self):
        self._reset()

    cdef inline bint _enough(self, AINDEX member_cutoff) nogil:
        if self.n_points > member_cutoff:
            return True
        return False

    def enough(self, member_cutoff: int):
        return self._enough(member_cutoff)

    cdef inline AINDEX _get_member(self, AINDEX index) nogil:
        cdef cppunordered_set[AINDEX].iterator it = self._neighbours.begin()
        cdef AINDEX i

        for i in range(index):
            preincrement(it)

        return dereference(it)

    def get_member(self, index: int):
        return self._get_member(index)

    cdef inline bint _contains(self, AINDEX member) nogil:
        if self._neighbours.find(member) == self._neighbours.end():
            return False
        return True

    def contains(self, member: int):
        return self._contains(member)


cdef class NeighboursExtVectorCPPUnorderedSet:
    """Implements the neighbours interface

    Uses a compination of an underlying C++ std:vector and a std:unordered_set.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a C++ vector.
    """

    def __cinit__(self, initial_size=1, neighbours=None):
        cdef AINDEX member

        self._initial_size = initial_size


        if neighbours is not None:
            self._neighbours = neighbours
            self.n_points = self._neighbours.size()
            self._neighbours.reserve(self._initial_size)

            for member in self._neighbours:
                self._neighbours_view.insert(member)
        else:
            self._reset()

    cdef inline void _assign(self, AINDEX member) nogil:
        self._neighbours.push_back(member)
        self._neighbours_view.insert(member)
        self.n_points += 1

    def assign(self, member: int):
        self._assign(member)

    cdef inline void _reset(self) nogil:
        self._neighbours.resize(0)
        self._neighbours.reserve(self._initial_size)
        self._neighbours_view.clear()
        self.n_points = 0

    def reset(self):
        self._reset()

    cdef inline bint _enough(self, AINDEX member_cutoff) nogil:
        if self.n_points > member_cutoff:
            return True
        return False

    def enough(self, member_cutoff: int):
        return self._enough(member_cutoff)

    cdef inline AINDEX _get_member(self, AINDEX index) nogil:
        return self._neighbours[index]

    def get_member(self, index: int):
        return self._get_member(index)

    cdef inline bint _contains(self, AINDEX member) nogil:
        if self._neighbours_view.find(member) == self._neighbours_view.end():
            return False
        return True

    def contains(self, member: int):
        return self._contains(member)


class NeighboursGetter(ABC):
    """Defines the neighbours-getter interface"""

    @property
    @abstractmethod
    def is_sorted(self) -> bool:
       """Return True if neighbour indices are sorted"""

    @property
    @abstractmethod
    def is_selfcounting(self) -> bool:
       """Return True if points count as their own neighbour"""

    @abstractmethod
    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours for point in input data"""

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours in input data for point in other input data"""


class NeighboursGetterBruteForce(NeighboursGetter):

    def __init__(self):
        self._is_sorted = False
        self._is_selfcounting = True

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    @property
    def is_selfcounting(self) -> bool:
        return self._is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i
        cdef AVALUE distance

        neighbours.reset()

        for i in range(input_data.n_points):
            distance = metric.calc_distance(index, i, input_data)

            if distance <= cluster_params.radius_cutoff:
                neighbours.assign(i)

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i
        cdef AVALUE distance

        neighbours.reset()

        for i in range(input_data.n_points):
            distance = metric.calc_distance_other(
                index, i, input_data, other_input_data
                )

            if distance <= cluster_params.radius_cutoff:
                neighbours.assign(i)


cdef class NeighboursGetterExtBruteForce:

    def __cinit__(self):
        self.is_sorted = False
        self.is_selfcounting = True

    cdef inline void _get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i
        cdef AVALUE distance

        neighbours._reset()

        for i in range(input_data.n_points):
            distance = metric._calc_distance(index, i, input_data)

            if distance <= cluster_params.radius_cutoff:
                neighbours._assign(i)

    def get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params):

        self._get(
            index,
            input_data,
            neighbours,
            metric,
            cluster_params,
        )

    cdef inline void _get_other(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i
        cdef AVALUE distance

        neighbours._reset()

        for i in range(input_data.n_points):
            distance = metric._calc_distance_other(
                index, i, input_data, other_input_data
                )

            if distance <= cluster_params.radius_cutoff:
                neighbours._assign(i)

    def get_other(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params):

        self._get_other(
            index,
            input_data,
            other_input_data,
            neighbours,
            metric,
            cluster_params,
        )


class NeighboursGetterLookup(NeighboursGetter):

    def __init__(self, is_sorted=False, is_selfcounting=False):
        self._is_sorted = is_sorted
        self._is_selfcounting = is_selfcounting

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    @property
    def is_selfcounting(self) -> bool:
        return self._is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:

        cdef AINDEX i

        neighbours.reset()

        for i in range(input_data.get_n_neighbours(index)):
            neighbours.assign(input_data.get_neighbour(index, i))

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i

        neighbours.reset()

        for i in range(other_input_data.get_n_neighbours(index)):
            neighbours.assign(other_input_data.get_neighbour(index, i))


cdef class NeighboursGetterExtLookup:

    def __cinit__(self, is_sorted=False, is_selfcounting=True):
        self.is_sorted = is_sorted
        self.is_selfcounting = is_selfcounting

    cdef inline void _get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i
        neighbours._reset()

        for i in range(input_data._get_n_neighbours(index)):
            neighbours._assign(input_data._get_neighbour(index, i))

    def get(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params):

        self._get(
            index,
            input_data,
            neighbours,
            metric,
            cluster_params,
        )

    cdef inline void _get_other(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i

        neighbours._reset()

        for i in range(other_input_data._get_n_neighbours(index)):
            neighbours._assign(other_input_data._get_neighbour(index, i))

    def get_other(
            self,
            AINDEX index,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data,
            NEIGHBOURS_EXT neighbours,
            METRIC_EXT metric,
            ClusterParameters cluster_params):

        self._get_other(
            index,
            input_data,
            other_input_data,
            neighbours,
            metric,
            cluster_params,
        )


class NeighboursGetterRecomputeLookup(NeighboursGetter):

    def __init__(self, is_sorted=False, is_selfcounting=False):
        self._is_sorted = is_sorted
        self._is_selfcounting = is_selfcounting

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    @property
    def is_selfcounting(self) -> bool:
        return self._is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:

        cdef AINDEX i

        if input_data._radius != cluster_params.radius_cutoff:
            input_data.compute_neighbourhoods(
                input_data,
                cluster_params.radius_cutoff,
                self._is_sorted,
                self._is_selfcounting
                )

        neighbours.reset()

        for i in range(input_data.get_n_neighbours(index)):
            neighbours.assign(input_data.get_neighbour(index, i))

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i

        if other_input_data._radius != cluster_params.radius_cutoff:
            other_input_data.compute_neighbourhoods(
                input_data,
                cluster_params.radius_cutoff,
                self._is_sorted,
                self._is_selfcounting
                )

        neighbours.reset()

        for i in range(other_input_data.get_n_neighbours(index)):
            neighbours.assign(other_input_data.get_neighbour(index, i))


class Metric(ABC):
    """Defines the metric-interface"""

    @abstractmethod
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        """Return distance between two points in input data"""

    @abstractmethod
    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:
        """Return distance between two points in input data and other input data"""


class MetricDummy(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        return 0.

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:
        return 0.

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff


cdef class MetricExtDummy:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        return 0.

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)

    cdef inline AVALUE _calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil:

        return 0.

    def calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) -> float:

        return self._calc_distance_other(
            index_a, index_b, input_data, other_input_data
            )

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return radius_cutoff

    def adjust_radius(self, radius_cutoff: float) -> float:
       return self._adjust_radius(radius_cutoff)


class MetricPrecomputed(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        return input_data.get_component(index_a, index_b)

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:

        return other_input_data.get_component(index_a, index_b)

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff


cdef class MetricExtPrecomputed:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        return input_data._get_component(index_a, index_b)

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)

    cdef inline AVALUE _calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil:

        return other_input_data._get_component(index_a, index_b)

    def calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) -> float:

        return self._calc_distance_other(
            index_a, index_b, input_data, other_input_data
            )

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return radius_cutoff

    def adjust_radius(self, radius_cutoff: float) -> float:
       return self._adjust_radius(radius_cutoff)


class MetricEuclidean(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff


cdef class MetricExtEuclidean:
    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)

    cdef inline AVALUE _calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) -> float:

        return self._calc_distance_other(
            index_a, index_b, input_data, other_input_data
            )

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return radius_cutoff

    def adjust_radius(self, radius_cutoff: float) -> float:
       return self._adjust_radius(radius_cutoff)


class MetricEuclideanReduced(Metric):

    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff**2


cdef class MetricExtEuclideanReduced:

    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)

    cdef inline AVALUE _calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) -> float:

        return self._calc_distance_other(
            index_a, index_b, input_data, other_input_data
            )

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return cpow(radius_cutoff, 2)

    def adjust_radius(self, radius_cutoff: float) -> float:
       return self._adjust_radius(radius_cutoff)


cdef class MetricExtEuclideanPeriodicReduced:

    def __cinit__(self, bounds):
        self._bounds = bounds

    cdef inline AVALUE _calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b, distance, bound

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)

            bound = self._bounds[i]
            distance = cfabs(a - b)

            if bound > 0:
                distance = distance % bound
                if distance > (bound / 2):
                    distance = bound - distance

            total +=  cpow(distance, 2)

        return total

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data) -> float:
        return self._calc_distance(index_a, index_b, input_data)

    cdef inline AVALUE _calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data.n_dim
        cdef AVALUE a, b, distance, bound

        for i in range(n_dim):
            a = other_input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)

            bound = self._bounds[i]
            distance = cfabs(a - b)

            if bound > 0:
                distance = distance % bound
                if distance > (bound / 2):
                    distance = bound - distance

            total += cpow(a - b, 2)

        return total

    def calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            INPUT_DATA_EXT input_data,
            INPUT_DATA_EXT other_input_data) -> float:

        return self._calc_distance_other(
            index_a, index_b, input_data, other_input_data
            )

    cdef inline AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return cpow(radius_cutoff, 2)

    def adjust_radius(self, radius_cutoff: float) -> float:
       return self._adjust_radius(radius_cutoff)


class SimilarityChecker(ABC):
    """Defines the similarity checker interface"""

    @abstractmethod
    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:
        """Retrun True if a and b have sufficiently many common neighbours"""


class SimilarityCheckerContains(SimilarityChecker):
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.
        The performance and time-complexity of the check depends on the
        used neighbour containers.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        no switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerSwitchContains`).
    """

    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:

        cdef AINDEX na = neighbours_a.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False


class SimilarityCheckerSwitchContains(SimilarityChecker):
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.  The performance
        and time-complexity of the check depends on the
        used neighbour containers.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that a
        switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerContains`).
    """

    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        if nb < na:
            neighbours_a, neighbours_b = neighbours_b, neighbours_a
            na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False


cdef class SimilarityCheckerExtContains:
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.
        The performance and time-complexity of the check depends on the
        used neighbour containers.
        Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        no switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerExtSwitchContains`).
    """

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX c = cluster_params.cnn_cutoff

        if c == 0:
            return True

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False

    def check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params):

        return self._check(neighbours_a, neighbours_b, cluster_params)


cdef class SimilarityCheckerExtSwitchContains:
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.
        The performance and time-complexity of the check depends on the
        used neighbour containers.
        Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`cnnclustering._types.SimilarityCheckerExtContains`).
    """

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil:


        cdef AINDEX c = cluster_params.cnn_cutoff

        if c == 0:
            return True

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if nb < na:
            with gil:
                neighbours_a, neighbours_b = neighbours_b, neighbours_a
                na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False

    def check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params):

        return self._check(neighbours_a, neighbours_b, cluster_params)


cdef class SimilarityCheckerExtScreensorted:
    r"""Implements the similarity checker interface

    Strategy:
        Loops over members of two neighbour containers alternatingly
        and checks if neighbours are contained in both containers.
        Requires that the containers are sorted ascendingly to return
        the correct result. Sorting will neither be checked nor enforced.
        Breaks
        early when similarity criterion is reached.
        The performance of the check depends on the
        used neighbour containers.
        Worst case time
        complexity is :math:`\mathcal{O}(n + m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers.
    """

    cdef inline bint _check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOUR_NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX c = cluster_params.cnn_cutoff

        if c == 0:
            return True

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        if (na == 0) or (nb == 0):
            return False

        cdef AINDEX member_index_a = 0, member_index_b = 0
        cdef AINDEX member_a, member_b
        cdef AINDEX common = 0

        member_a = neighbours_a._get_member(member_index_a)
        member_b = neighbours_b._get_member(member_index_b)

        while True:
            if member_a == member_b:
                common += 1
                if common == c:
                    return True

                member_index_a += 1
                member_index_b += 1

                if (member_index_a == na) or (member_index_b == nb):
                    break

                member_a = neighbours_a._get_member(member_index_a)
                member_b = neighbours_b._get_member(member_index_b)
                continue

            if member_a < member_b:
                member_index_a += 1
                if (member_index_a == na):
                    break
                member_a = neighbours_a._get_member(member_index_a)
                continue

            member_index_b += 1
            if (member_index_b == nb):
                break
            member_b = neighbours_b._get_member(member_index_b)

        return False


    def check(
            self,
            NEIGHBOURS_EXT neighbours_a,
            NEIGHBOURS_EXT neighbours_b,
            ClusterParameters cluster_params):

        return self._check(neighbours_a, neighbours_b, cluster_params)


class Queue(ABC):
    """Defines the queue interface"""

    @abstractmethod
    def push(self, value):
        """Put value into the queue"""

    @abstractmethod
    def pop(self):
        """Retrieve value from the queue"""

    @abstractmethod
    def is_empty(self):
        """Return True if there are no values in the queue"""


class QueueFIFODeque(Queue):
    """Implements the queue interface"""

    def __init__(self):
       self._queue = deque()

    def push(self, value):
        """Append value to back/right end"""
        self._queue.append(value)

    def pop(self):
        """Retrieve value from front/left end"""
        return self._queue.popleft()

    def is_empty(self):
        """Return True if there are no values in the queue"""
        if self._queue:
            return False
        return True


cdef class QueueExtLIFOVector:
    """Implements the queue interface"""

    cdef inline void _push(self, AINDEX value) nogil:
        """Append value to back/right end"""
        self._queue.push_back(value)

    cdef inline AINDEX _pop(self) nogil:
        """Retrieve value from back/right end"""

        cdef AINDEX value = self._queue.back()
        self._queue.pop_back()

        return value

    cdef inline bint _is_empty(self) nogil:
        """Return True if there are no values in the queue"""
        return self._queue.empty()

    def push(self, value: int):
        self._push(value)

    def pop(self) -> int:
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()


cdef class QueueExtFIFOQueue:
    """Implements the queue interface"""

    cdef inline void _push(self, AINDEX value) nogil:
        """Append value to back/right end"""
        self._queue.push(value)

    cdef inline AINDEX _pop(self) nogil:
        """Retrieve value from back/right end"""

        cdef AINDEX value = self._queue.front()
        self._queue.pop()

        return value

    cdef inline bint _is_empty(self) nogil:
        """Return True if there are no values in the queue"""
        return self._queue.empty()

    def push(self, value: int):
        self._push(value)

    def pop(self) -> int:
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()
