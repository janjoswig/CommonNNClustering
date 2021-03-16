from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from itertools import count
from typing import Any, Optional, Type

import numpy as np

from libc.math cimport sqrt as csqrt, pow as cpow

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class ClusterParameters:
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
    def shape(self):
        return self._labels.shape

    @property
    def consider_set(self):
        return self._consider_set

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
            _member_cutoff = 1
        else:
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
    def get_subset(self, indices: Sequence) -> Type['InputData']:
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

        if meta is None:
            meta = {}
        self._meta = meta

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def n_points(self):
        return len(self._data)

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def data(self):
        return [np.asarray(list(s)) for s in self._data]

    def get_component(self, point: int, dimension: int) -> float:
        """

        Method only present for consistency.
        Returns no relevant information.
        """
        return 0

    def get_n_neighbours(self, point: int) -> int:
        return self._n_neighbours[point]

    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""
        return self._data[point][member]

    def get_subset(self, indices: Sequence) -> Type['InputDataNeighboursSequence']:
        """Return input data subset"""
        data_subset = [
            [m for m in s if m in indices]
            for i, s in enumerate(self._data)
            if i in indices
        ]

        return type(self)(data_subset)


cdef class InputDataExtPointsMemoryview:
    """Implements the input data interface"""

    def __cinit__(self, AVALUE[:, ::1] data not None, *, meta=None):
        self._data = data
        self.n_points = self._data.shape[0]
        self.n_dim = self._data.shape[1]

        if meta is None:
            meta = {}
        self.meta = meta

    @property
    def data(self):
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

    def enough(self, member_cutoff: int):
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
    """Implements the neighbours interface"""

    def __cinit__(self, AINDEX initial_size, neighbours=None):
        self._initial_size = initial_size

        if neighbours is not None:
            self._neighbours = neighbours
            self.n_points = len(self._neighbours)
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
        cdef AINDEX index

        for index in range(self.n_points):
            if self._neighbours[index] == member:
                return True
        return False

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

        neighbours.reset()

        cdef AINDEX i

        for i in range(input_data.get_n_neighbours(index)):
            neighbours.assign(input_data.get_neighbour(index, i))


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


cdef class NeighboursGetterExtLookup:
    pass


class Metric(ABC):
    """Defines the metric-interface"""

    @abstractmethod
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        """Return distance between two points in input data"""


class MetricDummy(Metric):
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
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
        early when similarity criterion is reached.  Worst case time
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
        early when similarity criterion is reached.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
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
        early when similarity criterion is reached.  Worst case time
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

        cdef AINDEX na = neighbours_a.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

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
        early when similarity criterion is reached.  Worst case time
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

        cdef AINDEX na = neighbours_a.n_points
        cdef AINDEX nb = neighbours_b.n_points

        cdef AINDEX c = cluster_params.cnn_cutoff
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

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
