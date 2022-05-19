from abc import ABC, abstractmethod
from collections import deque
import copy
from typing import Any, Optional, Type, Union
from typing import Container, Iterable, List, Tuple, Sequence
import heapq
import weakref

import numpy as np

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from cnnclustering._types import (
    InputData,
    InputDataComponents,
    InputDataPairwiseDistances,
    InputDataPairwiseDistancesComputer,
    InputDataNeighbourhoods,
    InputDataNeighbourhoodsComputer,
    NeighboursGetter,
    Neighbours,
    DistanceGetter,
    Metric,
    SimilarityChecker,
    Queue,
    )


try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    NX_FOUND = False


from libcpp.vector cimport vector as cppvector
from libcpp.unordered_map cimport unordered_map as cppumap
from cython.operator cimport dereference, preincrement


class Fitter(ABC):
    """Defines the fitter interface"""

    @abstractmethod
    def fit(
            self,
            input_data: Type['InputData'],
            labels: Type['Labels'],
            cluster_params: Type['ClusterParameters']):
        """Generic clustering"""

    def make_parameters(
            self,
            radius_cutoff: float,
            cnn_cutoff: int,
            current_start: int) -> Type["ClusterParameters"]:

        try:
            used_metric = self._neighbours_getter._distance_getter._metric
        except AttributeError:
            pass
        else:
            radius_cutoff = used_metric.adjust_radius(radius_cutoff)

        n_member_cutoff = cnn_cutoff
        try:
            is_selfcounting = self._neighbours_getter.is_selfcounting
        except AttributeError:
            pass
        else:
            if is_selfcounting:
                n_member_cutoff += 1
                cnn_cutoff += 2

        cluster_params = ClusterParameters(
            radius_cutoff,
            cnn_cutoff,
            cnn_cutoff,  # similarity_cutoff_continuous not in use right now
            n_member_cutoff,
            current_start,
            )

        return cluster_params


class HierarchicalFitter(ABC):
    """Defines the hfitter interface"""

    @abstractmethod
    def fit(self, object clustering, **kwargs):
        """Generic clustering"""

    def make_parameters(
            self,
            radius_cutoff: float,
            cnn_cutoff: int,
            current_start: int) -> Type["ClusterParameters"]:

        try:
            used_metric = self._neighbours_getter._distance_getter._metric
        except AttributeError:
            pass
        else:
            radius_cutoff = used_metric.adjust_radius(radius_cutoff)

        n_member_cutoff = cnn_cutoff
        try:
            is_selfcounting = self._neighbours_getter.is_selfcounting
        except AttributeError:
            pass
        else:
            if is_selfcounting:
                n_member_cutoff += 1
                cnn_cutoff += 2

        cluster_params = ClusterParameters(
            radius_cutoff,
            cnn_cutoff,
            cnn_cutoff,  # similarity_cutoff_continuous not in use right now
            n_member_cutoff,
            current_start,
            )

        return cluster_params


cdef class FitterExtInterface:
    def make_parameters(
            self,
            radius_cutoff: float,
            cnn_cutoff: int,
            current_start: int) -> Type["ClusterParameters"]:

        try:
            used_metric = self._neighbours_getter._distance_getter._metric
        except AttributeError:
            pass
        else:
            radius_cutoff = used_metric.adjust_radius(radius_cutoff)

        n_member_cutoff = cnn_cutoff
        try:
            is_selfcounting = self._neighbours_getter.is_selfcounting
        except AttributeError:
            pass
        else:
            if is_selfcounting:
                n_member_cutoff += 1
                cnn_cutoff += 2

        cluster_params = ClusterParameters(
            radius_cutoff,
            cnn_cutoff,
            cnn_cutoff,  # similarity_cutoff_continuous not in use right now
            n_member_cutoff,
            current_start,
            )

        return cluster_params


class Predictor(ABC):
    """Defines the predictor interface"""

    @abstractmethod
    def predict(
            self,
            input_data: Type['InputData'],
            predictand_input_data: Type['InputData'],
            neighbours_getter: Type['NeighboursGetter'],
            predictand_neighbours_getter: Type['NeighboursGetter'],
            distance_getter: Type['DistanceGetter'],
            predictand_distance_getter: Type['DistanceGetter'],
            neighbours: Type['Neighbours'],
            neighbour_neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            similarity_checker: Type['SimilarityChecker'],
            labels: Type['Labels'],
            predictand_labels: Type['Labels'],
            cluster_params: Type['ClusterParameters']):
        """Generic cluster label prediction"""

    def make_parameters(
            self,
            radius_cutoff: float,
            cnn_cutoff: int,
            current_start: int) -> Type["ClusterParameters"]:

        try:
            used_metric = self._neighbours_getter._distance_getter._metric
        except AttributeError:
            pass
        else:
            radius_cutoff = used_metric.adjust_radius(radius_cutoff)

        n_member_cutoff = cnn_cutoff
        try:
            is_selfcounting = self._neighbours_getter.is_selfcounting
        except AttributeError:
            pass
        else:
            if is_selfcounting:
                n_member_cutoff += 1
                cnn_cutoff += 2

        cluster_params = ClusterParameters(
            radius_cutoff,
            cnn_cutoff,
            cnn_cutoff,  # similarity_cutoff_continuous not in use right now
            n_member_cutoff,
            current_start,
            )

        return cluster_params


class FitterBFS(Fitter):
    """Concrete implementation of the fitter interface

    Args:
        neighbours_getter: Any object implementing the neighbours getter
            interface.
        neighbours: Any object implementing the neighbours
            interface.
        neighbour_neighbourss: Any object implementing the neighbours
            interface.
        similarity_checker: Any object implementing the similarity checker
            interface.
        queue: Any object implementing the queue interface.
    """

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"],
            queue: Type["Queue"]):
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._queue = queue

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"queue={self._queue}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("queue", None),
            ]

    def fit(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params):
        """Generic common-nearest-neighbours clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`cnnclustering._types.Labels`.
            cluster_params: Instance of
                :obj:`cnnclustering._types.ClusterParameters`.
        """

        cdef AINDEX n_member_cutoff = cluster_params.n_member_cutoff
        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data.n_points
        current = cluster_params.current_start

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            self._neighbours_getter.get(
                init_point,
                input_data,
                self._neighbours,
                cluster_params
                )

            if not self._neighbours.enough(n_member_cutoff):
                continue

            _labels[init_point] = current

            while True:

                m = self._neighbours.n_points

                for member_index in range(m):
                    member = self._neighbours.get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter.get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    if not self._neighbour_neighbours.enough(n_member_cutoff):
                        _consider[member] = 0
                        continue

                    if self._similarity_checker.check(
                            self._neighbours,
                            self._neighbour_neighbours,
                            cluster_params):
                        _consider[member] = 0
                        _labels[member] = current
                        self._queue.push(member)

                if self._queue.is_empty():
                    break

                point = self._queue.pop()
                self._neighbours_getter.get(
                    point,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

            current += 1


cdef class FitterExtBFS(FitterExtInterface):
    """Concrete implementation of the fitter interface

    Args:
        neighbours_getter: Any extension type
            implementing the neighbours getter
            interface.
        neighbours: Any extension type implementing the neighbours
            interface.
        neighbour_neighbourss: Any extension type implementing the neighbours
            interface.
        similarity_checker: Any extension type implementing the similarity checker
            interface.
        queue: Any extension type implementing the queue interface. Used
            during the clustering procedure.
    """

    def __cinit__(
            self,
            NeighboursGetterExtInterface neighbours_getter,
            NeighboursExtInterface neighbours,
            NeighboursExtInterface neighbour_neighbours,
            SimilarityCheckerExtInterface similarity_checker,
            QueueExtInterface queue):
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._queue = queue

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"queue={self._queue}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("queue", None),
            ]

    cdef void _fit(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) nogil:
        """Generic common-nearest-neighbours clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`cnnclustering._types.Labels`.
            cluster_params: Instance of
                :obj:`cnnclustering._types.ClusterParameters`.
        """

        cdef AINDEX n_member_cutoff = cluster_params.n_member_cutoff
        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data.n_points
        current = cluster_params.current_start

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            self._neighbours_getter._get(
                init_point,
                input_data,
                self._neighbours,
                cluster_params
                )

            if not self._neighbours._enough(n_member_cutoff):
                continue

            _labels[init_point] = current

            while True:

                m = self._neighbours.n_points

                for member_index in range(m):
                    member = self._neighbours._get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter._get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    if not self._neighbour_neighbours._enough(n_member_cutoff):
                        _consider[member] = 0
                        continue

                    if self._similarity_checker._check(
                            self._neighbours,
                            self._neighbour_neighbours,
                            cluster_params):
                        _consider[member] = 0
                        _labels[member] = current
                        self._queue._push(member)

                if self._queue._is_empty():
                    break

                point = self._queue._pop()
                self._neighbours_getter._get(
                    point,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

            current += 1


    def fit(
        self,
        InputDataExtInterface input_data,
        Labels labels,
        ClusterParameters cluster_params):

        self._fit(
            input_data,
            labels,
            cluster_params,
            )


cdef class FitterExtBFSDebug(FitterExtInterface):
    """Concrete implementation of the fitter interface

    Yields/prints information during the clustering.

    Args:
        neighbours_getter: Any extension type
            implementing the neighbours getter
            interface.
        neighbours: Any extension type implementing the neighbours
            interface.
        neighbour_neighbourss: Any extension type implementing the neighbours
            interface.
        similarity_checker: Any extension type implementing the similarity checker
            interface.
        queue: Any extension type implementing the queue interface. Used
            during the clustering procedure.
    """

    def __cinit__(
            self,
            NeighboursGetterExtInterface neighbours_getter,
            NeighboursExtInterface neighbours,
            NeighboursExtInterface neighbour_neighbours,
            SimilarityCheckerExtInterface similarity_checker,
            QueueExtInterface queue,
            bint verbose=True,
            bint yielding=True):

        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._queue = queue
        self._verbose = verbose
        self._yielding = yielding

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"queue={self._queue}",
            f"verbose={self._verbose}",
            f"yielding={self._yielding}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("queue", None),
            ]

    def _fit(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:
        """Generic common-nearest-neighbours clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`cnnclustering._types.Labels`.
            cluster_params: Instance of
                :obj:`cnnclustering._types.ClusterParameters`.
        """

        cdef AINDEX n_member_cutoff = cluster_params.n_member_cutoff
        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data.n_points
        current = cluster_params.current_start

        if self._verbose:
            print(f"CommonNN clustering - {type(self).__name__}")
            print("=" * 80)
            print(f"{n} points")
            print(
                *(
                    f"{k:<29}: {v}"
                    for k, v in cluster_params.to_dict().items()
                ),
                sep="\n"
            )
            print()

        for init_point in range(n):

            if self._verbose:
                print(f"New source: {init_point}")

            if _consider[init_point] == 0:
                if self._verbose:
                    print("    ... already visited\n")
                continue
            _consider[init_point] = 0

            self._neighbours_getter._get(
                init_point,
                input_data,
                self._neighbours,
                cluster_params
                )

            if not self._neighbours._enough(n_member_cutoff):
                if self._verbose:
                    print("    ... not enough neighbours\n")
                continue

            _labels[init_point] = current
            if self._verbose:
                print(f"    ... new cluster {current}")

            if self._yielding:
                yield {
                    "reason": "assigned_source",
                    "init_point": init_point,
                    "point": None,
                    "member": None,
                    }

            while True:

                m = self._neighbours.n_points
                if self._verbose:
                    print(f"    ... loop over {m} neighbours")

                for member_index in range(m):
                    member = self._neighbours._get_member(member_index)

                    if self._verbose:
                        print(f"        ... current neighbour {member}")

                    if _consider[member] == 0:
                        if self._verbose:
                            print(f"        ... already visited\n")
                        continue

                    self._neighbours_getter._get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    if not self._neighbour_neighbours._enough(n_member_cutoff):
                        _consider[member] = 0
                        if self._verbose:
                            print("        ... not enough neighbours\n")
                        continue

                    if self._similarity_checker._check(
                            self._neighbours,
                            self._neighbour_neighbours,
                            cluster_params):

                        if self._verbose:
                            print("        ... successful check!\n")

                        _consider[member] = 0
                        _labels[member] = current

                        if self._yielding:
                            yield {
                                "reason": "assigned_neighbour",
                                "init_point": init_point,
                                "point": point,
                                "member": member,
                                }
                        self._queue._push(member)

                if self._queue._is_empty():
                    if self._verbose:
                        print("=" * 80)
                        print("end")
                    break

                point = self._queue._pop()

                if self._verbose:
                    print(f"    ... Next point: {point}")

                self._neighbours_getter._get(
                    point,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

            current += 1

    def fit(
        self,
        InputDataExtInterface input_data,
        Labels labels,
        ClusterParameters cluster_params):

        yield from self._fit(
            input_data,
            labels,
            cluster_params,
            )


Fitter.register(FitterExtBFS)
Fitter.register(FitterExtBFSDebug)


class HierarchicalFitterMSTPrim(HierarchicalFitter):
    """Concrete implementation of the fitter interface

    Args:
        neighbours_getter: Any object implementing the neighbours getter
            interface.
        neighbours: Any object implementing the neighbours
            interface.
        neighbour_neighbourss: Any object implementing the neighbours
            interface.
        similarity_checker: Any object implementing the similarity checker
            interface.
        priority_queue:
            Any object implementing the prioqueue interface (max heap).
        priority_queue_tree:
            Any object implementing the prioqueue interface (max heap).
    """

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"],
            priority_queue: Type["PriorityQueue"],
            priority_queue_tree: Type["PriorityQueue"]):
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._priority_queue = priority_queue
        self._priority_queue_tree = priority_queue_tree

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"prioq={self._priority_queue}",
            f"prioq (tree)={self._priority_queue_tree}"
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("priority_queue", None),
            ("priority_queue_tree", "priority_queue")
            ]

    def fit(self, object clustering, **kwargs):

        radius_cutoff = kwargs["radius_cutoff"]
        member_cutoff = kwargs.get("member_cutoff")
        max_clusters = kwargs.get("max_clusters")
        cnn_offset = kwargs.get("cnn_offset")
        sort_by_size = kwargs.get("sort_by_size", True)
        info = kwargs.get("info", True)
        v = kwargs.get("v", True)

        self._fit(
            clustering,
            radius_cutoff,
            member_cutoff,
            max_clusters,
            cnn_offset,
            sort_by_size,
            info,
            v
            )

    def _fit(
            self,
            bundle,
            radius_cutoff: float,
            member_cutoff: int = None,
            max_clusters: int = None,
            cnn_offset: int = None,
            sort_by_size: bool = True,
            info: bool = True,
            v: bool = True):

        self._priority_queue.reset()
        self._priority_queue_tree.reset()

        cdef object input_data = bundle._input_data
        cdef AINDEX n_points = input_data.n_points

        cdef object spanning_tree
        # TODO: Not actually needed (prioq tree is enough), but can be attached
        # TODO    to root bundle in the end for inspection

        cdef Labels labels = Labels(
            np.ones(n_points, order="C", dtype=P_AINDEX)
            )
        bundle._labels = labels
        cdef ABOOL* _consider = &labels._consider[0]

        cdef AINDEX m, member_index, a, b, i, j
        cdef AVALUE weight
        cdef AINDEX root, member, _member_cutoff

        cdef ClusterParameters cluster_params

        if cnn_offset is None:
            cnn_offset = 0

        cluster_params = self.make_parameters(
                radius_cutoff,
                0,
                1
                )

        if member_cutoff is None:
            member_cutoff = 10
        _member_cutoff = member_cutoff

        if not NX_FOUND:
            raise ModuleNotFoundError("No module named 'networkx'")

        # Build MST
        spanning_tree = nx.Graph()

        for root in range(n_points):
            if _consider[root] == 0:
                continue
            _consider[root] = 0
            spanning_tree.add_node(root)  # ! addition of nodes

            self._neighbours_getter.get(
                root,
                input_data,
                self._neighbours,
                cluster_params
                )

            m = self._neighbours.n_points

            for member_index in range(m):
                member = self._neighbours.get_member(member_index)

                if _consider[member] == 0:
                    continue

                self._neighbours_getter.get(
                    member,
                    input_data,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                weight = self._similarity_checker.get(
                    self._neighbours,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                self._priority_queue.push(root, member, weight)

            while not self._priority_queue.is_empty():
                a, b, weight = self._priority_queue.pop()

                if _consider[b] == 0:
                    continue
                _consider[b] = 0

                spanning_tree.add_edge(a, b, weight=weight)  # ! addition of w. edges
                self._priority_queue_tree.push(a, b, weight)

                self._neighbours_getter.get(
                    b,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

                m = self._neighbours.n_points

                for member_index in range(m):
                    member = self._neighbours.get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter.get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    weight = self._similarity_checker.get(
                        self._neighbours,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    self._priority_queue.push(b, member, weight)

        # Build hierarchy from MST
        cdef AVALUE merge_level, last_merge_level, new_max_weight
        cdef AINDEX n_edges = self._priority_queue_tree.size()
        cdef list union_find = []
        cdef AINDEX n_uf = -1
        cdef AINDEX tree_a, tree_b, members_a, members_b
        cdef dict children, children_a, children_b

        assert n_edges > 0, "No edges in the spanning tree"

        a, b, weight = self._priority_queue_tree.pop()
        merge_level = last_merge_level = weight

        union_find.append(
            Bundle(graph=nx.Graph([(a, b, {"weight": weight})]))
            )
        n_uf += 1
        union_find[n_uf].meta = {"max_weight": weight, "min_weight": None}

        for i in range(1, n_edges):
            a, b, weight = self._priority_queue_tree.pop()

            # Like Kruskal's
            tree_a = tree_b = -1
            for tree_index, x in enumerate(union_find):
                if a in x._graph:
                    tree_a = tree_index
                if b in x._graph:
                    tree_b = tree_index

            if (tree_a == -1) & (tree_b == -1):
                union_find.append(
                    Bundle(graph=nx.Graph([(a, b, {"weight": weight})]))
                    )
                n_uf += 1
                union_find[n_uf].meta = {"max_weight": weight, "min_weight": None}
            elif (tree_a == -1):
                union_find[tree_b]._graph.add_edge(a, b, weight=weight)
            elif (tree_b == -1):
                union_find[tree_a]._graph.add_edge(a, b, weight=weight)
            else:

                members_a = len(union_find[tree_a]._graph)
                members_b = len(union_find[tree_b]._graph)

                if (members_a >= _member_cutoff) & (members_b >= _member_cutoff):
                    if v:
                        print(f"Merge at {last_merge_level}/{weight}")

                    union_find[tree_a].meta["min_weight"] = last_merge_level
                    union_find[tree_b].meta["min_weight"] = last_merge_level

                    parent = Bundle(
                        graph=nx.union(union_find[tree_a]._graph, union_find[tree_b]._graph)
                        )
                    union_find.append(parent)
                    n_uf += 1
                    parent.meta = {"max_weight": weight, "min_weight": None}
                    union_find[tree_a]._parent = weakref.proxy(parent)
                    union_find[tree_b]._parent = weakref.proxy(parent)
                    parent._children[1] = union_find[tree_a]
                    parent._children[2] = union_find[tree_b]

                else:
                    children_a = union_find[tree_a]._children
                    children_b = union_find[tree_b]._children
                    assert (not children_a) | (not children_b), f"a: {children_a}, b: {children_b}"
                    # TODO: Remove assert if we are sure that this is really never violated.
                    # TODO     A cluster that has not enough members can not have children

                    if (members_a < _member_cutoff) & (members_b < _member_cutoff):
                        new_max_weight = weight
                        children = {}
                    elif members_a < _member_cutoff:
                        new_max_weight = union_find[tree_b].meta["max_weight"]
                        children = children_b
                    else: # elif members_b < member_cutoff:
                        new_max_weight = union_find[tree_a].meta["max_weight"]
                        children = children_a

                    union_find.append(
                        Bundle(graph=nx.union(union_find[tree_a]._graph, union_find[tree_b]._graph))
                        )
                    n_uf += 1
                    union_find[n_uf].meta = {
                        "max_weight": new_max_weight,
                        "min_weight": None,
                        }
                    union_find[n_uf]._children = children


                union_find[n_uf]._graph.add_edge(a, b, weight=weight)
                union_find = [
                    x for j, x in enumerate(union_find)
                    if (j != tree_a) & (j != tree_b)
                ]
                n_uf -= 2

            if weight < merge_level:
                last_merge_level = merge_level
            merge_level = weight

        i = 1
        noise = nx.Graph()
        for x in union_find:
            if len(x._graph) >= _member_cutoff:
                x.meta["min_weight"] = weight
                bundle._children[i] = x
                i += 1
            else:
                noise = nx.union(noise, x._graph)

        if len(noise) > 0:
            bundle._children[0] = Bundle(graph=noise)

        bundle._graph = spanning_tree


class HierarchicalFitterRepeat(HierarchicalFitter):

    def __init__(
            self,
            fitter: Type["Fitter"]):
        self._fitter = fitter

    def __str__(self):

        attr_str = ", ".join([
            f"fitter={self._fitter}"
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("fitter", None),
            ]

    def fit(self, Bundle bundle, **kwargs):

        radius_cutoff = kwargs["radius_cutoff"]
        cnn_cutoff = kwargs["cnn_cutoff"]
        member_cutoff = kwargs.get("member_cutoff")
        max_clusters = kwargs.get("max_clusters")
        cnn_offset = kwargs.get("cnn_offset")
        sort_by_size = kwargs.get("sort_by_size", True)
        info = kwargs.get("info", True)
        v = kwargs.get("v", True)

        self._fit(
            bundle,
            radius_cutoff,
            cnn_cutoff,
            member_cutoff,
            max_clusters,
            cnn_offset,
            sort_by_size,
            info,
            v
            )

    def _fit(
            self,
            bundle: Type["Bundle"],
            radius_cutoff: Union[float, Iterable[float]],
            cnn_cutoff: Union[int, Iterable[int]],
            member_cutoff: int = None,
            max_clusters: int = None,
            cnn_offset: int = None,
            sort_by_size: bool = True,
            info: bool = True,
            v: bool = True):

        cdef cppvector[AVALUE] radius_cutoff_vector
        cdef cppvector[AINDEX] cnn_cutoff_vector

        if not isinstance(radius_cutoff, Iterable):
            radius_cutoff = [radius_cutoff]

        radius_cutoff = [float(x) for x in radius_cutoff]

        if not isinstance(cnn_cutoff, Iterable):
            cnn_cutoff = [cnn_cutoff]

        cnn_cutoff = [int(x) for x in cnn_cutoff]

        if len(radius_cutoff) == 1:
            radius_cutoff *= len(cnn_cutoff)

        if len(cnn_cutoff) == 1:
            cnn_cutoff *= len(radius_cutoff)

        cdef AINDEX step, n_steps = len(radius_cutoff)
        assert n_steps == len(cnn_cutoff)

        radius_cutoff_vector = radius_cutoff
        cnn_cutoff_vector = cnn_cutoff

        cdef ClusterParameters cluster_params
        cdef AINDEX current_start = 1

        if cnn_offset is None:
            cnn_offset = 0

        cdef AINDEX n, n_points = bundle._input_data.n_points

        cdef Labels previous_labels = Labels(
            np.ones(n_points, order="C", dtype=P_AINDEX)
            )
        cdef Labels current_labels

        cdef AINDEX c_label, p_label

        cdef cppumap[AINDEX, cppvector[AINDEX]] parent_labels_map
        cdef cppumap[AINDEX, cppvector[AINDEX]].iterator p_it

        cdef dict terminal_clusterings = {1: bundle}
        cdef dict new_terminal_clusterings

        for step in range(n_steps):

            if v:
                print(
                    f"Running step {step:<5} "
                    f"(r = {radius_cutoff_vector[step]}, "
                    f"c = {cnn_cutoff_vector[step]})"
                    )

            new_terminal_clusterings = {}

            current_labels = Labels(
                np.zeros(n_points, order="C", dtype=P_AINDEX)
                )

            cluster_params = self.make_parameters(
                radius_cutoff_vector[step],
                cnn_cutoff_vector[step] - cnn_offset,
                current_start
                )

            self._fitter.fit(
                bundle._input_data,
                current_labels,
                cluster_params
                )

            if sort_by_size:
                current_labels.sort_by_size(member_cutoff, max_clusters)

            parent_labels_map.clear()

            for n in range(n_points):
                c_label = current_labels._labels[n]
                p_label = previous_labels._labels[n]

                if p_label == 0:
                    continue

                parent_labels_map[p_label].push_back(c_label)

            p_it = parent_labels_map.begin()
            while (p_it != parent_labels_map.end()):
                p_label = dereference(p_it).first

                # !!! Python interaction
                parent_clustering = terminal_clusterings[p_label]
                parent_clustering._labels = Labels.from_sequence(parent_labels_map[p_label])

                if info:
                    params = {
                        k: (radius_cutoff_vector[step], cnn_cutoff_vector[step])
                        for k in parent_clustering._labels.to_set()
                        if k != 0
                        }
                    parent_clustering._labels.meta.update({
                        "params": params,
                        "reference": weakref.proxy(parent_clustering),
                        "origin": "fit"
                    })

                parent_clustering.isolate(isolate_input_data=False)

                for c_label, child_clustering in parent_clustering._children.items():
                    if c_label == 0:
                        continue
                    new_terminal_clusterings[c_label] = child_clustering

                preincrement(p_it)

            terminal_clusterings = new_terminal_clusterings
            previous_labels = current_labels


HierarchicalFitter.register(HierarchicalFitterMSTPrim)
HierarchicalFitter.register(HierarchicalFitterRepeat)


class PredictorFirstmatch(Predictor):

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours_getter_other: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"]):
        self._neighbours_getter = neighbours_getter
        self._neighbours_getter_other = neighbours_getter_other
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"ngetter_other={self._neighbours_getter_other}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours_getter_other", "neighbours_getter"),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ]

    def predict(
            self,
            object input_data,
            object predictand_input_data,
            Labels labels,
            Labels predictand_labels,
            ClusterParameters cluster_params):
        """Generic cluster label prediction"""

        cdef AINDEX n_member_cutoff = cluster_params.n_member_cutoff
        cdef AINDEX n, m, point, member, member_index, label

        cdef AINDEX* _labels = &labels._labels[0]
        cdef AINDEX* _predictand_labels = &predictand_labels._labels[0]
        cdef ABOOL* _consider = &predictand_labels._consider[0]
        cdef cppunordered_set[AINDEX] _consider_set = predictand_labels._consider_set

        n = predictand_input_data.n_points

        for point in range(n):
            if _consider[point] == 0:
                continue

            self._neighbours_getter_other.get_other(
                point,
                input_data,
                predictand_input_data,
                self._neighbours,
                cluster_params
            )

            if not self._neighbours.enough(n_member_cutoff):
                continue

            m = self._neighbours.n_points
            for member_index in range(m):
                member = self._neighbours.get_member(member_index)
                label = _labels[member]

                if _consider_set.find(label) == _consider_set.end():
                    continue

                self._neighbours_getter.get(
                    member,
                    input_data,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                if not self._neighbour_neighbours.enough(n_member_cutoff):
                    continue

                if self._similarity_checker.check(
                        self._neighbours,
                        self._neighbour_neighbours,
                        cluster_params):
                    _consider[point] = 0
                    _predictand_labels[point] = label
                    break

        return
