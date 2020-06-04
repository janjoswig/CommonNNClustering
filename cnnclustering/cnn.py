#!/usr/bin/env python

"""cnn - A Python module for common-nearest-neighbour (CNN) clustering

The functionality provided in this module is based on code implemented
by Oliver Lemke in the script collection CNNClustering available on
git-hub (https://github.com/BDGSoftware/CNNClustering.git). Please cite:

    * B. Keller, X. Daura, W. F. van Gunsteren J. Chem. Phys.,
        2010, 132, 074110.
    * O. Lemke, B.G. Keller, J. Chem. Phys., 2016, 145, 164104.
    * O. Lemke, B.G. Keller, Algorithms, 2018, 11, 19.
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, namedtuple, UserList
from collections.abc import MutableSequence
import functools
from itertools import count
import pickle
from pathlib import Path
import random
# import sys
import tempfile
import time
import warnings
from typing import Dict, List, Set, Tuple
from typing import Collection, Iterator, Sequence  # Iterable
from typing import Any, Optional, Type, Union, IO

import colorama  # TODO Make this optional or remove completely?
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    # Optional dependency
    import pandas as pd
    _PANDAS_FOUND = True
except ModuleNotFoundError:
    print("Did not load pandas")
    _PANDAS_FOUND = False

from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import tqdm
import yaml

from . import _cfits
# from . import _conversions
from . import _fits
from . import _plots


def timed(function_):
    """Decorator to measure execution time.

    Forwards the output of the wrapped function and measured execution
    time as a tuple.
    """

    @functools.wraps(function_)
    def wrapper(*args, **kwargs):
        go = time.time()
        wrapped = function_(*args, **kwargs)
        stop = time.time()
        if wrapped is not None:
            stopped = stop - go
            hours, rest = divmod(stopped, 3600)
            minutes, seconds = divmod(rest, 60)
            print(
                "Execution time for call of "
                f"{function_.__name__}: "
                f"{int(hours)} hours, "
                f"{int(minutes)} minutes, "
                f"{seconds:.4f} seconds"
            )
            return *wrapped, stopped
        return
    return wrapper


def recorded(function_):
    """Decorator to format fit function feedback.

    Used to decorate fit methods of `CNN` instances.  Feedback needs to
    be sequence in record format, i.e. conforming to the `CNNRecord`
    namedtuple.  If execution time was measured, the corresponding field
    will be modified.
    """

    @functools.wraps(function_)
    def wrapper(self, *args, **kwargs):
        wrapped = function_(self, *args, **kwargs)
        if wrapped is not None:
            if len(wrapped) == 3:
                record = wrapped[0]._replace(time=wrapped[-1])
            else:
                record = wrapped[0]

            if wrapped[1]:
                # Be chatty
                print("-" * 80)
                print(
                    "#points   ",
                    "R         ", "N         ", "M         ",
                    "max       ", "#clusters ", "%largest  ", "%noise    ",
                    sep="")
                for entry in record[:-1]:
                    if entry is None:
                        print(f"{'None':<10}", end="")
                    elif isinstance(entry, float):
                        print(f"{entry:<10.3f}", end="")
                    else:
                        print(f"{entry:<10}", end="")
                print("\n" + "-" * 80)

            self.summary.append(record)

        return
    return wrapper


class SparsegraphArray(np.ndarray):
    """Sparse graph representation of density connectivity
    """

    def __new__(
            cls,
            edges: Optional[Sequence[int]] = None,
            indices: Optional[Sequence[int]] = None):

        if edges is None:
            edges = []

        if indices is None:
            indices = []

        obj = np.asarray(edges, dtype=np.uintp).view(cls)
        obj._indices = np.asarray(indices, dtype=np.uintp)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._indices = getattr(obj, '_indices', None)

    @property
    def indicesx(self):
        return self._indices


class Labels(np.ndarray):
    """Cluster label assignments

    Attributes:
        info: Instance of `LabelInfo` metadata.
        consider: Boolean array of same length as labels, indicating
            which labels should be still considered (e.g. for
            predictions)
    """

    def __new__(
            cls, sequence: Optional[Sequence[int]] = None,
            info=None, consider=None):
        if sequence is None:
            sequence = []

        obj = np.asarray(sequence, dtype=np.int_).view(cls)
        obj.info = info

        if consider is None:
            consider = np.ones_like(obj, dtype=np.uint8)

        obj._consider = consider

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.info = getattr(obj, 'info', None)
        self._consider = getattr(obj, '_consider', None)

        # TODO Warning in this place does not work, e.g. makes np.all()
        #    on labels fail ...
        # for i in self:
        #    if i < 0:
        #         warnings.resetwarnings()
        #         warnings.warn(
        #             "Passed sequence contains negative elements. "
        #             "Labels should be positiv integers or 0 (noise).",
        #             UserWarning
        #             )
        #         sys.stderr.flush()
        #         break

    @property
    def consider(self):
        return self._consider

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, x):
        if x is None:
            self._info = LabelInfo(
                origin=None,
                reference=None,
                params={},
                )
            return

        if not isinstance(x, LabelInfo):
            raise TypeError(
                "Label information must be of type `LabelInfo` or `None`"
                )

        self._info = x

    @property
    def clusterdict(self):
        return self.labels2dict(self)

    @staticmethod
    def labels2dict(labels: Collection[int]) -> Dict[int, Set[int]]:
        """Convert labels to cluster dictionary

        Args:
           labels: Sequence of integer cluster labels to convert

        Returns:
           Dictionary of sets of point indices with cluster labels
              as keys
        """

        dict_ = defaultdict(set)
        for index, l in enumerate(labels):
            dict_[l].add(index)

        return dict_

    @staticmethod
    def dict2labels(
            dictionary: Dict[int, Collection[int]]) -> Type[np.ndarray]:
        """Convert cluster dictionary to labels

        Args:
            dictionary: Dictionary of point indices per cluster label
                to convert

        Returns:
            Sequenc of labels for each point as NumPy ndarray
        """

        labels = np.zeros(
            np.sum(len(x) for x in dictionary.values())
            )

        for key, value in dictionary.items():
            labels[value] = key

        return labels

    def fix_missing(self):
        """Fix missing cluster labels and ensure continuous numbering

        If you also want the labels to be sorted by clustersize use
        :meth:`sort_by_size` instead, which re-numbers clusters, too.
        """

        assert np.all(self >= 0)

        # fixing  missing labels
        ulabels = set(self)
        n_clusters = len(ulabels)
        if 0 not in ulabels:
            n_clusters += 1

        d = 0  # Total number of missing labels
        for c in range(1, n_clusters):
            # Next label continuous?
            if (c + d) in ulabels:
                continue

            # Gap of missing labels
            next_greater = c + 1
            while (next_greater + d) not in ulabels:
                next_greater += 1

            # Correct label numbers downwards
            d_ = next_greater - c
            self[self > c] -= d_
            d += d_  # Keep track of missing labels in total

    def sort_by_size(
            self, member_cutoff=None, max_clusters=None):
        """Sort labels by clustersize in-place

        Re-assigns cluster numbers so that the biggest cluster (that is
        not noise) is cluster 1.  Also filters clusters out, that have
        not at least `member_cutoff` members.  Optionally, does only
        keep the `max_clusters` largest clusters.  Returns the member
        count in the largest cluster and the number of points declared
        as noise.

        Args:
           member_cutoff: Valid clusters need to have at least this
              many members.
           max_clusters: Only keep this many clusters.

        Returns:
           (#member largest, #member noise)
        """

        # Check that labels are not negative
        for i in self:
            if i < 0:
                raise ValueError(
                    "Passed sequence contains negative elements. "
                    "Labels should be positiv integers or 0 (noise)."
                    )

        if member_cutoff is None:
            member_cutoff = int(settings.get(
                "default_member_cutoff",
                settings.defaults.get("default_member_cutoff")
                ))

        frequencies = Counter(self)
        if 0 in frequencies:
            _ = frequencies.pop(0)

        if frequencies:
            order = frequencies.most_common()
            reassign = {}
            reassign[0] = 0

            new_labels = count(1)
            for pair in order:
                if pair[1] >= member_cutoff:
                    reassign[pair[0]] = next(new_labels)
                else:
                    reassign[pair[0]] = 0

            if max_clusters is not None:
                for key in reassign:
                    if key > max_clusters:
                        reassign[key] = 0

            for index, old_label in enumerate(self):
                self[index] = reassign[old_label]

            if self.info.params:
                new_params = {
                    reassign[k]: v
                    for k, v in self.info.params.items()
                    if reassign[k] != 0
                    }
                self.info = self.info._replace(params=new_params)

        return

    def merge(self, clusters: List[int]) -> None:
        """Merge a list of clusters into one"""

        if len(clusters) < 2:
            raise ValueError(
                "List of clusters needs to have at least 2 elements"
                )

        if not isinstance(clusters, list):
            clusters = list(clusters)
        clusters.sort()

        base = clusters[0]

        for add in clusters[1:]:
            self[self == add] = base

        self.sort_by_size()

        return

    def trash(self, clusters: List[int]) -> None:
        """Merge a list of clusters into noise"""

        for remove in clusters:
            self[self == remove] = 0

        self.sort_by_size()

        return


class NeighbourhoodsABC(ABC):
    """Abstraction class for neighbourhoods

    Neighbourhoods (integer point indices) can be stored in different
    data structures (non-exhaustive listing):

        Collection of collections:
            - list of sets
            - array of arrays

        Linear collection plus slice indicator:
            - array of neighbours plus array of starting indices
            - array of neighbours plus array of lengths
            - array in which one element indicates the length and the
                following elements are neighbours

    To qualify as a neighbourhoods container, the following attributes
    should be also present in any case:

        radius: Points are neighbours of each other with respect to
            this radius (any metric)
        reference: A `CNN` instance, if neighbourhoods are valid for
            points in different data sets
        n_neighbours: Return the neighbourcount for each point in
            the container
        __str__: A useful str-representation revealing the type and
            the radius
    """

    @abstractmethod
    def __str__(self):
        """Reveal type of neighbourhoods and radius"""

    @property
    @abstractmethod
    def radius(self):
        """Return radius"""

    @radius.setter
    @abstractmethod
    def radius(self, x):
        """Ensure integrity and set radius"""

    @property
    @abstractmethod
    def n_neighbours(self):
        """Return number of neighbours for each point"""

    @property
    @abstractmethod
    def reference(self):
        """Return reference CNN instance"""

    @reference.setter
    @abstractmethod
    def reference(self, x):
        """Ensure integrity and set reference CNN instance"""


class Neighbourhoods(NeighbourhoodsABC):
    """Basic realisation of neighbourhood abstraction

    Makes no assumptions on the nature of the stored neighbours and
    provides default implementations for the required attributes by
    `NeighboursABC`. Since working realisations of the `Neighbourhoods`
    base class usually inherit with priority from a collection type
    whose `__init__` mechanism is probably used, the alternative method
    `init_finalise` is offered to set the required attributes.
    """

    def __init__(self, neighbourhoods=None, radius=None, reference=None):
        self.neighbourhoods = neighbourhoods
        self.init_finalise(radius=radius, reference=reference)

    def init_finalise(self, radius=None, reference=None):
        self.radius = radius
        self.reference = reference

    def __str__(self):
        return f"Neighbourhoods, radius = {self.radius}"

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, x):
        if x is not None:
            x = float(x)
        self._radius = x

    @property
    def n_neighbours(self):
        [len(x) for x in self.neighbourhoods]

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, x):
        if x is None:
            self._reference = None
            return

        if not isinstance(x, CNN):
            raise ValueError(
                "Reference should be of type `CNN` or `None`"
                )

        self._reference = x


class NeighbourhoodsList(UserList, Neighbourhoods):

    def __init__(self, neighbourhoods=None, radius=None, reference=None):
        if neighbourhoods is None:
            neighbourhoods = []
        super().__init__(neighbourhoods)
        super().init_finalise(radius=radius, reference=reference)

    @property
    def n_neighbours(self):
        [len(x) for x in self]


class NeighbourhoodsArray(np.ndarray, Neighbourhoods):

    def __init__(
            self, sequence: Optional[Sequence[int]] = None,
            radius=None, reference=None):
        pass

    def __new__(
            cls, sequence: Optional[Sequence[int]] = None,
            radius=None, reference=None):
        if sequence is None:
            sequence = []

        obj = np.asarray(sequence, dtype=object).view(cls)
        obj.radius = radius
        obj.reference = reference

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        super().init_finalise(
            radius=getattr(obj, 'radius', None),
            reference=getattr(obj, 'reference', None)
            )

    @property
    def n_neighbours(self):
        [x.shape[0] for x in self]


class NeighbourhoodsLinear(np.ndarray, Neighbourhoods):
    """Sparse graph representation of neighbourhoods

    Elements are neighbour counts of a specific point followed by
    elements that are indices of neighbouring points.
    """

    def __init__(
            self, sequence: Optional[Sequence[int]] = None,
            radius=None, reference=None):
        pass

    def __new__(
            cls, sequence: Optional[Sequence[int]] = None,
            radius=None, reference=None):
        if sequence is None:
            sequence = []

        obj = np.asarray(sequence, dtype=int).view(cls)
        obj.radius = radius
        obj.reference = reference

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        super().init_finalise(
            radius=getattr(obj, 'radius', None),
            reference=getattr(obj, 'reference', None)
            )

    @property
    def n_neighbours(self):
        if self.shape[0] == 0:
            return []

        n = []  # Output
        i = 0
        while True:
            try:
                n_ = self[i]
            except IndexError:
                # Reached the end
                try:
                    # Check if neighbours and indicators are consistent
                    _ = self[i - 1]
                except IndexError as e:
                    raise RuntimeError(
                        "There is something wrong in your neighbourhood. "
                        "No, seriously: The neighbour count indicators "
                        "do not add up properly!"
                        ) from e
                else:
                    break
            else:
                n.append(n_)
                i += n_ + 1
        return n


class Data:
    """Abstraction class for handling input data

    Bundles points, distances, neighbourhoods and
    auxillaries like trees.
    """

    # TODO Add refindex here?

    def __init__(
            self,
            points=None,
            distances=None,
            neighbourhoods=None,
            graph=None):
        self.points = points
        self.distances = distances
        self.neighbourhoods = neighbourhoods
        self.graph = graph

    @property
    def shape(self):
        shapes = {
            "points": self.points.shape[0],
            "distances": self.distances.shape[0],
            "neighbourhoods": len(self.neighbourhoods),
            }
        shape_set = set(shapes.values())
        if len(shape_set) == 1:
            return tuple(shape_set)

        shape_set.discard(0)
        if len(shape_set) == 1:
            return tuple(shape_set)

        raise RuntimeError(f"Inconsistent data shapes: {shapes}")

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, x):
        if not isinstance(x, Points):
            x = Points.from_parts(x)
        self._points = x
        # TODO This is the right point for choosing the constructor

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, x):
        if not isinstance(x, Distances):
            x = Distances(x)
        self._distances = x

    @property
    def neighbourhoods(self):
        return self._neighbourhoods

    @neighbourhoods.setter
    def neighbourhoods(self, x):
        if x is None:
            x = NeighbourhoodsList()

        if not isinstance(x, NeighbourhoodsABC):
            raise TypeError
            # Choose converter
            # x = Neighbourhoods(x)
        self._neighbourhoods = x


class Distances(np.ndarray):
    """Abstraction class for data points


    """

    def __new__(
            cls,
            p: Optional[np.ndarray] = None,
            edges: Optional[Sequence] = None,
            reference=None):
        if p is None:
            p = np.array([])
        obj = p.view(cls)
        obj._edges = edges
        obj._reference = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._edges = getattr(obj, "edges", None)
        self._reference = getattr(obj, "reference", None)

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, x):
        sum_edges = sum(x)
        if (self.shape[0] != 0) and (sum_edges != self.shape[0]):
            warnings.warn(
                f"Part edges ({sum(x)} points) do not match data points "
                f"({self.shape[0]} points)"
                )
        self._edges = x

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, x):
        if x is None:
            self._reference = None
            return

        if not isinstance(x, CNN):
            print(type(x))
            print(issubclass(type(x), CNN))
            raise ValueError(
                "Reference should be of type `CNN`"
                )

        self._reference = x


class Points(np.ndarray):
    """Abstraction class for data points

    """

    def __new__(
            cls,
            p: Optional[np.ndarray] = None,
            edges: Optional[Sequence] = None,
            tree: Optional[Any] = None):
        if p is None:
            p = np.array([])
        obj = p.view(cls)
        obj._edges = edges
        obj._tree = tree
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._edges = getattr(obj, "edges", None)
        self._tree = getattr(obj, "tree", None)

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, x):
        sum_edges = sum(x)
        if (self.shape[0] != 0) and (sum_edges != self.shape[0]):
            warnings.warn(
                f"Part edges ({sum(x)} points) do not match data points "
                f"({self.shape[0]} points)"
                )
        self._edges = x

    @property
    def tree(self):
        return self._tree

    @classmethod
    def from_parts(cls, p: Optional[Sequence]):
        """Alternative constructor

        Use if data is passed as collection of parts, as
            >>> obj = Points.from_parts([[[0],[1]], [[2],[3]]])

        Recognised input formats are:
            * Sequence
            * 2D Sequence (sequence of sequences all of same length)
            * Sequence of 2D sequences all of same second dimension

        In this way, part edges are taken from the input shape and do
        not have to be specified explicitly. Calls :meth:`get_shape`.

        Args:
            p: File name as string or :obj:`pathlib.Path` object

        Return:
            Instance of :obj:`Points`
        """

        return cls(*cls.get_shape(p))

    @classmethod
    def from_file(
            cls,
            f: Union[str, Path], *args, from_parts: bool = False,
            **kwargs):
        """Alternative constructor

        Use if data is passed as collection of parts, as
            >>> obj = Points.from_parts([[[0],[1]], [[2],[3]]])

        Recognised input formats are:
            * Sequence
            * 2D Sequence (sequence of sequences all of same length)
            * Sequence of 2D sequences all of same second dimension

        In this way, part edges are taken from the input shape and do
        not have to be specified explicitly. Calls :meth:`get_shape`
        and :meth:`load`.

        Args:
            f: File name as string or :obj:`pathlib.Path` object

        Return:
            Instance of :obj:`Points`
        """

        if from_parts:
            return cls(*cls.get_shape(cls.load(f, *args, **kwargs)))
        return cls(cls.load(f, *args, **kwargs))

    @staticmethod
    def get_shape(data: Any):
        """Maintain data in universal shape

        Analyses the format of given data and fits it into the standard
        format (parts, points, dimensions).  Creates a
        :obj:`numpy.ndarray` vstacked along the parts componenent that
        can be passed to the `Points` constructor along part edges.
        This may not be able to deal with all possible kinds of input
        structures correctly, so check the outcome carefully.

        Recognised input formats are:
            * Sequence
            * 2D Sequence (sequence of sequences all of same length)
            * Sequence of 2D sequences all of same second dimension

        Args:
            data: Either None or
                * a 1D sequence of length x,
                    interpreted as 1 point in x dimension
                * a 2D sequence of length x (rows) times y (columns),
                    interpreted as x points in y dimension
                * a list of 2D sequences,
                    interpreted as groups (parts) of points

        Returns:
            data -- A numpy.ndarray of shape (sum points, dimension)
            edges -- Part edges (list), marking the end points of the
                parts
        """

        if data is None:
            return None, None

        data_shape = np.shape(data[0])
        # raises a type error if data is not subscribable

        if np.shape(data_shape)[0] == 0:
            # 1D Sequence passed
            data = [np.array([data])]

        elif np.shape(data_shape)[0] == 1:
            # 2D Sequence of sequences passed"
            assert len({len(s) for s in data}) == 1
            data = [np.asarray(data)]

            # TODO Does not catch edge case in which sequence of 2D
            #    sequences is passed, not all having the same dimension

        elif np.shape(data_shape)[0] == 2:
            assert len({len(s) for s_ in data for s in s_}) == 1
            data = [np.asarray(x) for x in data]

        else:
            raise ValueError(
                f"Data shape {data_shape} not allowed"
                )

        edges = [x.shape[0] for x in data]

        return np.vstack(data), edges

    def by_parts(self) -> Iterator:
        """Yield data by parts

        Returns:
            Generator of 2D :obj:`numpy.ndarray`s (parts)
        """

        if self.size > 0:
            start = 0
            for end in self.edges:
                yield self[start:(start + end), :]
                start += end

        else:
            yield from ()

    @staticmethod
    def load(f: Union[Path, str], *args, **kwargs) -> None:
        """Loads file content

        Depending on the filename extension, a suitable loader is
        called:

            * .p: :func:`pickle.load`
            * .npy: :func:`numpy.load`
            * None: :func:`numpy.loadtxt`
            * .xvg, .dat: :func:`numpy.loadtxt`

        Sets :attr:`data` and :attr:`shape`.

        Args:
            f: File
            *args: Passed to loader

        Keyword Args:
            **kwargs: Passed to loader

        Returns:
            Return value of the loader
        """

        extension = Path(f).suffix

        case_ = {
            '.p': lambda: pickle.load(
                open(f, 'rb'),
                *args,
                **kwargs
                ),
            '.npy': lambda: np.load(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
            '': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
            '.xvg': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
            '.dat': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
             }

        return case_.get(
            extension,
            lambda: print(f"Unknown filename extension {extension}")
            )()

    def cKDtree(self, **kwargs):
        """Wrapper for `scipy.spatial.cKDTree`

        Sets `Points.tree`.

        Args:
            **kwargs: Passed to `scipy.spatial.cKDTree`
        """

        self._tree = cKDTree(self, **kwargs)


class Summary(MutableSequence):
    def __init__(self, iterable=None):
        if iterable is None:
            iterable = ()
        self._list = list(iterable)

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, item):
        if isinstance(item, CNNRecord):
            self._list.__setitem__(key, item)
        else:
            raise TypeError(
                "Summary can only contain records of type `CNNRecord`"
            )

    def __delitem__(self, key):
        self._list.__delitem__(key)
        # trigger change handler

    def __len__(self):
        return self._list.__len__()

    def __str__(self):
        return self._list.__str__()

    def insert(self, index, item):
        if isinstance(item, CNNRecord):
            self._list.insert(index, item)
        else:
            raise TypeError(
                "Summary can only contain records of type `CNNRecord`"
            )

    def to_DataFrame(self):
        """Convert list of records to (typed) pandas.DataFrame"""

        if not _PANDAS_FOUND:
            raise ModuleNotFoundError("Did not load pandas")

        _record_dtypes = [
            pd.Int64Dtype(),  # points
            np.float64,       # r
            pd.Int64Dtype(),  # n
            pd.Int64Dtype(),  # m
            pd.Int64Dtype(),  # max
            pd.Int64Dtype(),  # clusters
            np.float64,       # largest
            np.float64,       # noise
            np.float64,       # time
            ]

        content = []
        for field in CNNRecord._fields:
            content.append([
                record.__getattribute__(field)
                for record in self._list
                ])

        return TypedDataFrame(
            columns=CNNRecord._fields,
            dtypes=_record_dtypes,
            content=content,
            )

    def summarize(
            self,
            ax: Optional[Type[mpl.axes.SubplotBase]] = None,
            quant: str = "time",
            treat_nan: Optional[Any] = None,
            ax_props: Optional[Dict] = None,
            contour_props: Optional[Dict] = None):
        """Generate a 2D plot of record values

        Record values ("time", "clusters", "largest", "noise") are
        plotted against cluster parameters (radius cutoff "r"
        and cnn cutoff "n").

        Args:
            ax: `Axes` instance.  If `None`, a new `Figure` with `Axes`
                will be created.
            quant: Record value to visualise.
                * "time"
                * "clusters"
                * "largest"
                * "noise"
            treat_nan: If not `None`, use this value to pad nan-values.
            ax_props:
            contour_props:
        """

        if len(self._list) == 0:
            raise LookupError(
                "No cluster result records in summary"
                )

        ax_props_defaults = {
            "xlabel": "$R$",
            "ylabel": "$N$",
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

        plotted = _plots.plot_summary(
            ax, self.to_DataFrame(),
            quant=quant,
            treat_nan=treat_nan,
            contour_props=contour_props_defaults
            )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted


LabelInfo = namedtuple(
    'LabelInfo', [
            "origin",     # "fitted", "reeled", "predicted", None
            "reference",  # another CNN instance, None
            "params",     # dict of fit/predict params per label
        ]
    )

CNNRecord = namedtuple(
    'CNNRecord', [
        "points",
        "r",
        "n",
        "m",
        "max",
        "clusters",
        "largest",
        "noise",
        "time",
        ]
    )


class CNN:
    """CNN cluster object class

    A cluster object connects input data (points, distances, neighbours,
    ...) to cluster labels via clustering methodologies (fits).

    Parameters:
        points:
        distances:
        neighbourhoods:
        labels:
        alias:

    Attributes:
        labels
    """

    def __init__(
            self,
            points: Optional[Any] = None,
            distances: Optional[Any] = None,
            neighbourhoods: Optional[Any] = None,
            labels: Collection[int] = None,
            alias: str = "root") -> None:

        self.alias = alias        # Descriptive object identifier
        self.hierarchy_level = 0  # See hierarchy_level.setter

        self.data = Data(
            points, distances, neighbourhoods
            )

        self.labels = labels  # See labels.setter
        self.summary = Summary()
        self._children = None
        self._refindex = None
        self._refindex_rel = None
        self._tree = None
        self._status = None

    @property
    def hierarchy_level(self):
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, level):
        self._hierarchy_level = int(level)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, sequence: Optional[Sequence[int]]):
        self._labels = Labels(sequence)

    @property
    def children(self):
        return self._children

    @property
    def refindex(self):
        return self._refindex

    @property
    def refindex_rel(self):
        return self._refindex_rel

    @property
    def status(self):
        """Current data situation"""

        self.check()
        return self._status

    @staticmethod
    def check_present(attribute):
        if attribute is not None:
            return True
        return False

    def check(self):
        """Check current data state

        Check if data points, distances or neighbourhoods are present
        and in which format.  Check depends on length of the stored
        objects.
        """

        self._status = {}

        # Check for data points
        if len(self.data.points) > 0:
            self._status["points"] = (True, self.data.points.shape[0])
        else:
            self._status["points"] = (False,)

        if len(self.data.points.edges) > 0:
            self._status["edges"] = (True, len(self.data.points.edges))
        else:
            self._status["edges"] = (False,)

        # Check for point distances
        if len(self.data.distances) > 0:
            self._status["distances"] = (True, self.data.distances.shape[0])
        else:
            self._status["distances"] = (False,)

        # Check for neighbourhoods
        if len(self.data.neighbourhoods) > 0:
            self._status["neighbourhoods"] = (True,
                                              len(self.data.neighbourhoods),
                                              self.data.neighbourhoods.radius)
        else:
            self._status["neighbourhoods"] = (False,)

        # TODO Check for graph

    def __str__(self):

        # Check data situation
        self.check()

        if self._status["edges"][0]:
            if self._status['edges'][1] > 1:
                if self._status['edges'][1] < 5:
                    edge_str = (f"{self._status['edges'][1]}, "
                                f"{self.data.points.edges}")
                else:
                    edge_str = (f"{self._status['edges'][1]}, "
                                f"{self.data.points.edges[:5]}")
            else:
                edge_str = f"{self._status['edges'][1]}"
        else:
            edge_str = "None"

        if self._status["points"][0]:
            points_str = f"{self._status['points'][1]}"
            dim_str = f"{self.data.points.shape[1]}"
        else:
            points_str = "None"
            dim_str = "None"

        if self._status["distances"][0]:
            dist_str = f"{self._status['distances'][1]}"
        else:
            dist_str = "None"

        if self._status["neighbourhoods"][0]:
            neigh_str = (f"{self._status['neighbourhoods'][1]}, "
                         f"r = {self._status['neighbourhoods'][2]}")
        else:
            neigh_str = "None"

        str_ = (
            f'{"=" * 80}\n'
            "CNN cluster object\n"
            f'{"-" * 80}\n'
            f"alias :{' ' * 25}{self.alias}\n"
            f"hierachy level :{' ' * 16}{self.hierarchy_level}\n"
            "\n"
            f"data point shape :{' ' * 14}Parts      - {edge_str}\n"
            f"{' ' * 32}Points     - {points_str}\n"
            f"{' ' * 32}Dimensions - {dim_str}\n"
            "\n"
            f"distance matrix calculated :{' ' * 4}{dist_str}\n"
            f"neighbour list calculated :{' ' * 5}{neigh_str}\n"
            f"clustered :{' ' * 21}{self.labels.size > 0}\n"
            f"children :{' ' * 22}{self.check_present(self._children)}\n"
            f'{"=" * 80}\n'
            )

        return str_

    def cut(
            self,
            part: Optional[int] = None,
            points: Tuple[Optional[int], ...] = None,
            dimensions: Tuple[Optional[int], ...] = None):
        """Create a new :obj:`CNN` instance from a data subset

        Convenience function to create a reduced cluster object.
        Supported are continuous slices from the original data that
        allow making a view instead of a copy.

        Args:
            part: Cut out the points for exactly one part
                (zero based index).
            points: Slice points by using (start:stop:step)
            dimensions: Slice dimensions by using (start:stop:step)
        """

        # TODO Implement part

        if points is None:
            points = (None, None, None)

        if dimensions is None:
            dimensions = (None, None, None)

        if self.data.points.size > 0:
            _points = self.data.points[slice(*points), slice(*dimensions)]

        # TODO Implement cut distance matrix

        return type(self)(points=_points)

    @timed
    def calc_dist(
            self,
            other=None,
            v: bool = True,
            method: str = 'cdist',
            mmap: bool = False,
            mmap_file: Optional[Union[Path, str, IO[bytes]]] = None,
            chunksize: int = 10000, progress: bool = True):
        """Computes a distance matrix (points x points)

        Accesses data points in given data of standard shape

        Args:
            other: If not `None`, a second :obj:`CNN` cluster object.
                Distances to points in this associated with this object
                will be calculated.
            v: Be chatty
            method: Method to compute distances
                * cdist: :func:`Use scipy.spatial.distance.cdist`
            mmap: Wether to memory map the calculated distances on disk
                (NumPy)
            mmap_file: If `mmap` is set to True, where to store the
                file.  If None, uses a temporary file.
            chunksize: Portions of data to process at once.  Can be used
                to keep memory consumption low.  Only useful together
                with `mmap`.
            progress: Wether to show a progress bar.
        """

        if self.data.points.size == 0:
            return

        progress = not progress  # TODO Keep Progressbars?

        if other is None:
            other = self

        if method == 'cdist':
            if mmap:
                if mmap_file is None:
                    mmap_file = tempfile.TemporaryFile()

                len_self = self.data.points.shape[0]
                len_other = other.data.points.shape[0]
                self.data.distances = np.memmap(
                    mmap_file,
                    dtype=settings.float_precision_map[
                        settings["float_precision"]
                        ],
                    mode='w+',
                    shape=(len_self, len_other),
                    )
                chunks = np.ceil(len_self / chunksize).astype(int)
                for chunk in tqdm.tqdm(
                        range(chunks), desc="Mapping",
                        disable=progress, unit="Chunks", unit_scale=True,
                        bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (
                            colorama.Style.BRIGHT,
                            colorama.Fore.BLUE,
                            colorama.Fore.RESET
                            )):
                    self.data.distances[
                            chunk*chunksize: (chunk+1)*chunksize] = cdist(
                        self.data.points[chunk*chunksize: (chunk+1)*chunksize],
                        self.data.points
                        )
            else:
                self.data.distances = cdist(self.data.points,
                                            other.data.points)

            self.data.distances.reference = other

        else:
            raise ValueError(
                f"Method {method} not understood."
                "Currently implemented methods:\n"
                "    'cdist'"
                )

    def calc_neighbours_from_dist(self, r: float, format="array_arrays"):
        """Calculate neighbour list at a given radius

        Requires :attr:`self.data.distances`.
        Sets :attr:`self.data.neighbourhoods`.

        Args:
            r: Radius cutoff

        Returns:
            None
        """

        if format == "list_sets":
            neighbourhoods = [
                set(np.where((x > 0) & (x < r))[0])
                for x in self.data.distances
                ]

            self.data.neighbourhoods = NeighbourhoodsList(neighbourhoods, r)
        elif format == "array_arrays":
            neighbourhoods = np.array([
                np.where((x > 0) & (x < r))[0].astype(np.intp)
                for x in self.data.distances
                ])

            self.data.neighbourhoods = NeighbourhoodsArray(neighbourhoods, r)

        self.data.neighbourhoods.reference = self.data.distances.reference

    def calc_neighbours_from_cKDTree(
            self, r: float, other=None, **kwargs):
        """Wrapper for :meth:`scipy.spatial.cKDTree.query_ball_tree`

        Requires :attr:`self.data.points.tree`
        Sets :attr:`self.data.neighbourhoods`

        Args:
            r: Search radius
            other: If not `None`, another :obj:`CNN` instance whose data
               points should be used for relative neighbour search.
               Also requires  :attr:`other.data.points.tree`.
            **kwargs: Keyword args passed on to
               :meth:`scipy.spatial.cKDTree.query_ball_tree`
        """
        assert self.data.points.tree is not None

        if other is None:
            other = self

        neighbourhoods = [
            set(x)
            for x in self.data.points.tree.query_ball_tree(
                other.data.points.tree,
                r, **kwargs
                )
            ]

        self.data.neighbourhoods = NeighbourhoodsList(neighbourhoods, r)
        self.data.neighbourhoods.reference = other

    def dist_hist(
            self,
            ax: Optional[Type[mpl.axes.SubplotBase]] = None,
            maxima: bool = False,
            maxima_props: Optional[Dict[str, Any]] = None,
            hist_props: Optional[Dict[str, Any]] = None,
            ax_props: Optional[Dict[str, Any]] = None,
            inter_props: Optional[Dict[str, Any]] = None):
        """Plot a histogram of distances in the data set

        Requires distances to be pre-computed.

        Args:
            ax: `Axes` to plot on. If `None`, `Figure` and `Axes` are
                created.
            maxima: Whether to mark the maxima of the
                distribution. Uses :func:`scipy.signal.argrelextrema`.
            maxima_props: Keyword arguments passed to
               :func:`scipy.signal.argrelextrema` if `maxima` is set
               to True.
            maxima_props: Keyword arguments passed to
                :func:`numpy.histogram` to compute the histogram.
            ax_props: Keyword arguments passed
                to :func:`ax.set` for styling.
        """

        # TODO Move to distances class / _plots.py

        # TODO Add option for kernel density estimation
        # (scipy.stats.gaussian_kde, statsmodels.nonparametric.kde)

        if self.data.distances is None:
            raise ValueError(
                "No distances calculated."
                )

        # TODO make this a configuration option
        hist_props_defaults = {
            "bins": 100,
            "density": True,
        }

        if hist_props is not None:
            hist_props_defaults.update(hist_props)

        histogram, bins = np.histogram(
            self.data.distances.flat,
            **hist_props_defaults
            )

        binmids = 0.5 * (bins[:-1] + bins[1:])

        if inter_props is not None:
            # TODO make this a configuation option
            inter_props_defaults = {
                "ifactor": 0.5,
                "kind": 'linear',
            }

            inter_props_defaults.update(inter_props)

            ifactor = inter_props_defaults.pop("ifactor")

            ipoints = int(
                np.ceil(len(binmids) * ifactor)
                )
            ibinmids = np.linspace(binmids[0], binmids[-1], ipoints)
            histogram = interp1d(
                binmids,
                histogram,
                **inter_props_defaults
                )(ibinmids)

            binmids = ibinmids

        ylimit = np.max(histogram) * 1.1

        # TODO make this a configuration option
        ax_props_defaults = {
            "xlabel": "d / au",
            "ylabel": '',
            "yticks": (),
            "xlim": (np.min(binmids), np.max(binmids)),
            "ylim": (0, ylimit),
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        line = ax.plot(binmids, histogram)

        if maxima:
            maxima_props_ = {
                "order": 2,
                "mode": "clip"
                }

            if maxima_props is not None:
                maxima_props_.update(maxima_props)

            found = argrelextrema(histogram, np.greater, **maxima_props_)[0]
            settings['default_radius_cutoff'] = \
                f"{binmids[found[0]]:.2f}"

            annotations = []
            for candidate in found:
                annotations.append(
                    ax.annotate(
                        f"{binmids[candidate]:.2f}",
                        xy=(binmids[candidate], histogram[candidate]),
                        xytext=(binmids[candidate],
                                histogram[candidate] + (ylimit / 100))
                        )
                    )
        else:
            annotations = None

        ax.set(**ax_props_defaults)

        return fig, ax, line, annotations

    @recorded
    @timed
    def fit(
            self,
            radius_cutoff: Optional[float] = None,
            cnn_cutoff: Optional[int] = None,
            member_cutoff: Optional[int] = None,
            max_clusters: Optional[int] = None,
            cnn_offset: Optional[int] = None,
            info: bool = True,
            sort_by_size: bool = True,
            rec: bool = True, v: bool = True,
            policy: str = "progressive",
            ) -> Optional[Tuple[CNNRecord, bool]]:
        """Wraps CNN clustering execution

        This function prepares the clustering and calls an appropriate
        worker function to do the actual clustering.  How the clustering
        is done, depends on the current data situation and the selected
        `policy`. The clustering can be done either with data points,
        pre-computed pairwise point distances, or pre-computed
        neighbourhoods as input.  Ultimately, neighbourhoods are used
        during the clustering.  Clustering is fast if neighbourhoods are
        pre-computed but this has to be done for each `radius_cutoff`
        separately. Neighbourhoods can be calculated either from data
        points, or pre-computed pairwaise distances.  Storage of
        distances can be costly memory-wise.  If the user chooses
        `policy = "progressive"`, neighbourhoods will be computed from
        either distances (if present) or points before the clustering.
        If the user chooses `policy = "conservative"`, neighbourhoods
        will be computed on-the-fly (online) from either distances (if
        present) or points during the clustering.  This can save memory
        but can be computational more expensive.  Caching can be used to
        achieve the right balance between memory usage and computing
        effort for your situation.
        """

        assert policy in ["progressive", "conservative"]

        # Set params
        param_template = {
            # option name, (user option name, used as type here)
            'radius_cutoff': (radius_cutoff, float),
            'cnn_cutoff': (cnn_cutoff, int),
            'member_cutoff': (member_cutoff, int),
            'cnn_offset': (cnn_offset, int),
            }

        params = {}

        for option, (value, type_) in param_template.items():
            if value is None:
                default = f"default_{option}"
                params[option] = type_(settings.get(
                    default, settings.defaults.get(default)
                    ))
            else:
                params[option] = param_template[option][1](
                    param_template[option][0]
                    )

        params["cnn_cutoff"] -= params["cnn_offset"]
        assert params["cnn_cutoff"] >= 0

        # Check data situation
        self.check()

        # Neighbourhoods calculated?
        if (self._status["neighbourhoods"][0] and
                self.data.neighbourhoods.radius == params["radius_cutoff"]):
            # Fit from pre-computed neighbourhoods,
            #     no matter what the policy is
            self.labels = Labels(np.zeros(len(self.data.neighbourhoods),
                                          dtype=np.int_))
            fit_fxn = _cfits.fit_from_NeighbourhoodsArray
            fit_args = (self.data.neighbourhoods,
                        self.labels,
                        self.labels.consider,
                        params["cnn_cutoff"])
            # TODO: Allow different methods and data structures

        # Distances calculated?
        elif self._status["distances"][0]:
            if policy == "progressive":
                # Pre-compute neighbourhoods from distances
                self.calc_neighbours_from_dist(params["radius_cutoff"])
                self.labels = Labels(np.zeros(self.data.distances.shape[0],
                                              dtype=np.int_))
                fit_fxn = _cfits.fit_from_NeighbourhoodsArray
                fit_args = (self.data.neighbourhoods,
                            self.labels,
                            self.labels.consider,
                            params["cnn_cutoff"])

            elif policy == "conservative":
                # Use distances as input and calculate neighbours online
                raise NotImplementedError()

        # Points loaded?
        elif self._status["points"][0]:
            if policy == "progressive":
                # Pre-compute neighbourhoods from points
                raise NotImplementedError()
            elif policy == "conservative":
                # Use points as input and calculate neighbours online
                fit_fxn = _cfits.fit_from_PointsArray
                self.labels = Labels(np.zeros(self.data.points.shape[0],
                                              dtype=np.int_))
                fit_args = (self.data.points,
                            self.labels,
                            self.labels.consider,
                            params["radius_cutoff"],
                            params["cnn_cutoff"],
                            )
        else:
            raise LookupError(
                "No input data (neighbours, distances, or points) found"
                )

        # Call clustering
        fit_fxn(*fit_args)  # Modify self.labels in-place

        if sort_by_size:
            # Sort by size and filter
            self.labels.sort_by_size(
                member_cutoff=params["member_cutoff"],
                max_clusters=max_clusters,
                )

        if info:
            # Attach info
            self.labels.info = LabelInfo(
                origin="fitted",
                reference=self,
                params={
                    k: (params["radius_cutoff"], params["cnn_cutoff"])
                    for k in self.labels.clusterdict
                    if k != 0
                    },
                )

        if rec:
            noise = 0
            frequencies = Counter(self.labels)
            if 0 in frequencies:
                noise = frequencies.pop(0)

            largest = frequencies.most_common(1)[0][1] if frequencies else 0

            return CNNRecord(
                self.data.points.shape[0],
                # TODO Maintain rather on Data level
                params["radius_cutoff"],
                params["cnn_cutoff"],
                params["member_cutoff"],
                max_clusters,
                self.labels.max(),
                largest / self.data.points.shape[0],
                noise / self.data.points.shape[0],
                None,
                ), v

        return None

    @timed
    def predict(
            self,
            other,
            radius_cutoff: Optional[float] = None,
            cnn_cutoff: Optional[int] = None,
            # member_cutoff: Optional[int] = None,
            include_all: bool = True,
            same_tol: float = 1e-8,
            memorize: bool = True,
            clusters: Optional[List[int]] = None,
            purge: bool = False,
            cnn_offset: Optional[int] = None,
            behaviour: str = "lookup",
            method: str = 'plain',
            progress: bool = True,
            policy="progressive",
            **kwargs
            ) -> None:
        """
        Predict labels for points in a data set (`other`) on the basis
        of assigned labels to a "train" set (`self`).

        Args:
            other: `CNN` cluster object for whose points cluster labels
                should be predicted

            radius_cutoff: Find nearest neighbours within
                distance *r*

            cnn_cutoff: Points of the same cluster must have
                at least *n* common nearest neighbours
                (Similarity criterion)

            member_cutoff: Clusters must have more than *m* members or
                are declared noise

            include_all:
                If `False`, keep cluster assignment for points in the test set
                that have a maximum distance of `same_tol` to a point
                in the train set, i.e. they are (essentially the same point)
                (currently not implemented)

            same_tol: Distance cutoff to treat points as the same, if
                `include_all` is `False`

            clusters: Predict assignment of points only with respect to
                this list of clusters

            purge: If `True`, reinitalise predicted labels.
                Override assignment memory.

            memorize:  # TODO obsolet? Always true if purge false?
                If `True`, remember which points in the test set have
                been already assigned and exclude them from future
                predictions

            cnn_offset: Mainly for backwards compatibility.
                Modifies the the cnn_cutoff.

            behaviour : str, default="lookup"
                Controlls how the predictor operates:

                * "lookup", Use distance matrices CNN.train_distances and
                    CNN.map_matrix to lookup distances to generate the
                    neighbour lists.  If one of the matrices does not exist,
                    throw an error.  Consider memory mapping `mmap`
                    when computing the distances with :py:meth:`CNN.dist` and
                    :py:meth:`CNN.map` for large data sets.

                * "on-the-fly", Compute distances during the prediction
                    using the specified `method`.

                * "tree", Get the neighbour lists during the prediction from
                    a tree query

            method : str, default="plain"
                Controlls which method is used to get the neighbour lists
                within a given `behaviour`:

                * "lookup", parameter not used

                * "on-the-fly",
                    * "plain", uses :py:meth:`CNN.get_neighbours`

                * "tree", parameter not used

            progress : bool, default=True
                Show a progress bar

            **kwargs :
                Additional keyword arguments are passed to the method that
                is used to compute the neighbour lists
        """

        ################################################################
        # TODO: Implement include_all mechanism.  The current version
        #     acts like include_all = True.
        ################################################################

        # Set params
        param_template = {
            # option name, (user option name, used as type here)
            'radius_cutoff': (radius_cutoff, float),
            'cnn_cutoff': (cnn_cutoff, int),
            # 'member_cutoff': (member_cutoff, int),
            'cnn_offset': (cnn_offset, int),
            }

        params = {}

        for option, (value, type_) in param_template.items():
            if value is None:
                default = f"default_{option}"
                params[option] = type_(settings.get(
                    default, settings.defaults.get(default)
                    ))
            else:
                params[option] = param_template[option][1](
                    param_template[option][0]
                    )

        params["cnn_cutoff"] -= params["cnn_offset"]
        assert params["cnn_cutoff"] >= 0

        # Check data situation
        self.check()
        other.check()

        # TODO: Decouple memorize?
        if purge or (clusters is None):
            other.labels = np.zeros(other.data.shape[0]).astype(int)
            if clusters is None:
                clusters = list(self.labels.clusterdict.keys())

        else:
            if other.labels.size == 0:
                other.labels = np.zeros(other.data.shape[0]).astype(int)

            for cluster_ in clusters:
                other.labels[other.labels == cluster_] = 0

        # Neighbourhoods calculated?
        if (other._status["neighbourhoods"][0] and
                other.data.neighbourhoods.radius == params["radius_cutoff"]):
            # Fit from pre-computed neighbourhoods,
            # no matter what the policy is
            predict_fxn = _fits.predict_from_neighbours
            predict_args = (params["cnn_cutoff"],
                            other.data.neighbourhoods,
                            other.labels,
                            other.labels._consider,
                            self.labels,
                            set(clusters))

            # Predict from List[Set[int]]
            # TODO: Allow different methods and data structures

        # Distances calculated?
        elif other._status["distances"][0]:
            if policy == "progressive":
                # Pre-compute neighbourhoods from distances
                other.calc_neighbours_from_dist(r=params["radius_cutoff"])
                predict_fxn = _fits.predict_from_neighbours
                predict_args = (params["cnn_cutoff"],
                                other.data.neighbourhoods,
                                other.labels,
                                other.labels._consider,
                                self.labels,
                                set(clusters))

            elif policy == "conservative":
                # Use distances as input and calculate neighbours online
                raise NotImplementedError()

        # Points loaded?
        elif other._status["points"][0]:
            if policy == "progressive":
                # Pre-compute neighbourhoods from points
                raise NotImplementedError()
            elif policy == "conservative":
                # Use points as input and calculate neighbours online
                raise NotImplementedError()
        else:
            raise LookupError(
                "No input data (neighbours, distances, or points) found"
                )

        # Call prediction
        predict_fxn(*predict_args)

        # Attach info
        other.labels.info = other.labels.info._replace(
            origin="predicted",
            reference=other
            )
        for cluster_ in clusters:
            other.labels.info.params[cluster_] = (params["radius_cutoff"],
                                                  params["cnn_cutoff"])

    @staticmethod
    def check_similarity_sequence(
            a: Sequence[int], b: Sequence[int], c: int) -> bool:
        """Check if similarity criterion is fulfilled.

        Args:
            a: Sequence of point indices
            b: Sequence of point indices
            c: Similarity cut-off

        Returns:
            True if sequence `a` and sequence `b` have at least `c` common
            elements
        """

        if len(set(a).intersection(b)) >= c:
            return True
        return False

    @staticmethod
    def check_similarity_set(a: Set[int], b: Set[int], c: int) -> bool:
        """Check if similarity criterion is fulfilled.

        Args:
            a: Sequence of point indices
            b: Sequence of point indices
            c: Similarity cut-off

        Returns:
            True if set `a` and set `b` have at least `c` common
            elements
        """

        if len(a.intersection(b)) >= c:
            return True
        return False

    @staticmethod
    def check_similarity_array(
            a: Type[np.ndarray], b: Type[np.ndarray], c: int) -> bool:
        """Check if similarity criterion is fulfilled for NumPy arrays.

        Args:
            a: Array of point indices
            b: Array of point indices
            c: Similarity cut-off

        Returns:
            True if array `a` and array `b` have at least `c` common
            elements
        """

        if len(np.intersect1d(a, b, assume_unique=True)) >= c:
            return True
        return False

    def isolate(self, purge: bool = True) -> None:
        """Isolates points per clusters based on a cluster result"""

        if purge or self._children is None:
            self._children = defaultdict(lambda: CNNChild(self))

        for label, cpoints in self.labels.clusterdict.items():

            cpoints = list(cpoints)
            # cpoints should be a sorted list due to the way clusterdict
            # is constructed by :meth:`Labels.labels2dict`

            ref_index = []           # point indices in root
            ref_index_relative = []  # point indices in parent

            if self._refindex is None:
                # Isolate from root
                ref_index.extend(cpoints)
                ref_index_relative = ref_index
            else:
                # Isolate from child
                ref_index.extend(self._refindex[cpoints])
                ref_index_relative.extend(cpoints)

            # TODO Extent isolate to work with distances
            #    Work with neighbourhoods probably does not make sense
            self._children[label].data.points = self.data.points[cpoints]
            # copies data from parent to child

            child_edges = None
            # Should have the same length as in root in the end
            # (or be None)
            if self.data.points.edges is not None:
                # Pass on part edges to child
                child_edges = []
                edges = iter(self.data.points.edges)
                part_end = next(edges)
                # End index of points in the first part
                child_e_ = 0
                # Number of points in the first part going to the child

                for index in ref_index:
                    if index < part_end:
                        child_e_ += 1
                        continue

                    while index >= part_end:
                        child_edges.append(child_e_)
                        part_end += next(edges)
                        # End index of points in the next part
                        child_e_ = 0
                        # Reset number of points in this part
                        # going to the child

                    child_e_ += 1
                child_edges.append(child_e_)

            self._children[label].data.points.edges = child_edges
            self._children[label]._refindex = np.asarray(ref_index)
            self._children[label]._refindex_relative = np.asarray(
                ref_index_relative)
            self._children[label].alias = f'child No. {label}'

        return

    def reel(self, deep: Optional[int] = 1) -> None:
        """Wrap up assigments of lower hierarchy levels

        Args:
            deep: How many lower levels to consider.  If `None`,
                consider all.
        """

        def reel_children(parent, deep):
            if parent._children is None:
                return

            if deep is not None:
                deep -= 1

            parent.labels.info = parent.labels.info._replace(origin="reeled")

            for c, child in parent._children.items():
                if (deep is None) or (deep > 0):
                    reel_children(child, deep)  # Dive deeper

                n_clusters = max(parent.labels)

                if child.labels.size > 0:
                    # Child has been clustered
                    for index, label in enumerate(child.labels):
                        if label == 0:
                            new_label = 0
                        else:
                            new_label = label + n_clusters
                        parent.labels[child._refindex_relative[index]] = \
                            new_label

                    if c in parent.labels.info.params:
                        del parent.labels.info.params[c]

                    if child.labels.info:
                        for label, p in child.labels.info.params.items():
                            parent.labels.info.params[label + n_clusters] = \
                                p

        if deep is not None:
            assert deep > 0

        reel_children(self, deep)

        return

    def get_samples(
            self, kind: str = 'mean', clusters: Optional[List[int]] = None,
            n_samples: int = 1, byparts: bool = True,
            skip: int = 0, stride: int = 1) -> Dict[int, List[int]]:
        """Select sample points from clusters

        Args:
            kind: How to choose the samples
                * "mean":
                * "random":
                * "all":
            clusters: List of cluster numbers to consider
            n_samples: How many samples to return
            byparts: Return point indices as list of lists by parts
            skip: Skip the first *n* frames
            stride: Take only every *n*th frame

        Returns:
            Samples -- Dictionary of sample point indices as list for
                each cluster
        """

        dict_ = self._clusterdict
        _data = np.vstack(self._data)
        _shape = self._shape

        if clusters is None:
            clusters = list(range(1, len(dict_)+1))

        samples = defaultdict(list)
        if kind == 'mean':
            for cluster in clusters:
                points = dict_[cluster]
                mean = np.mean(_data[points], axis=0)
                sumofsquares = np.sum((_data - mean)**2, axis=1)
                include = np.ones(len(_data), dtype=bool)
                n_samples_ = min(n_samples, len(points))
                for s in range(n_samples_):
                    least = sumofsquares[include].argmin()
                    samples[cluster].extend(least)
                    include[least] = False

        elif kind == 'random':
            for cluster in clusters:
                points = np.asarray(dict_[cluster])
                n_samples_ = min(n_samples, len(points))
                samples[cluster].extend(
                    points[
                        np.asarray(random.sample(
                            range(0, len(points)), n_samples
                            ))
                        ]
                    )

        elif kind == 'all':
            for cluster in clusters:
                samples[cluster].extend(
                    np.asarray(dict_[cluster])[::stride]
                    )

        else:
            raise ValueError()

        if byparts:
            part_borders = np.cumsum(_shape['points'])
            for cluster, points in samples.items():
                pointsbyparts = []
                for point in points:
                    part = np.searchsorted(part_borders, point)
                    if part > 0:
                        point = (point - part_borders[part - 1]) * skip
                    pointsbyparts.append(
                        (part, point)
                        )
                samples[cluster] = pointsbyparts

        return samples

    def get_dtraj(self):
        """Transform cluster labels to discrete trajectories

        Returns:
            [type]: [description]
        """

        _dtrajs = []

        _shape = self._shape
        _labels = self._labels

        # TODO: Better use numpy split()?
        part_startpoint = 0
        for part in range(0, _shape['parts']):
            part_endpoint = part_startpoint \
                + _shape['points'][part]

            _dtrajs.append(_labels[part_startpoint: part_endpoint])

            part_startpoint = np.copy(part_endpoint)

        return _dtrajs

    def evaluate(
            self,
            ax: Optional[Type[mpl.axes.SubplotBase]] = None,
            clusters: Optional[Collection[int]] = None,
            original: bool = False,
            plot: str = 'dots',
            # TODO parts: Optional[Tuple[Optional[int]]] = None,
            points: Optional[Tuple[Optional[int]]] = None,
            dim: Optional[Tuple[int, int]] = None,
            ax_props: Optional[Dict] = None, annotate: bool = True,
            annotate_pos: str = "mean",
            annotate_props: Optional[Dict] = None,
            plot_props: Optional[Dict] = None,
            plot_noise_props: Optional[Dict] = None,
            hist_props: Optional[Dict] = None,
            free_energy: bool = True,
            # TODO mask: Optional[Sequence[Union[bool, int]]] = None,
            ):

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

            plot:
                The kind of plotting method to use.

                    * "dots", :func:`ax.plot`
                    * "scatter", :func:`ax.scatter`
                    * "contour", :func:`ax.contour`
                    * "contourf", :func:`ax.contourf`

            parts:
                Use a slice (start, stop, stride) on the data parts
                before plotting.

            points:
                Use a slice (start, stop, stride) on the data points
                before plotting.

            dim:
                Use these two dimensions for plotting.  If `None`, uses
                (0, 1).

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
                functions (:func:`_plots.plot_dots` etc.) with different
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

            mask:
                Sequence of boolean or integer values used for optional
                fancy indexing on the point data array.  Note, that this
                is applied after regular slicing (e.g. via `points`) and
                requires a copy of the indexed data (may be slow and
                memory intensive for big data sets).

        Returns:
            Figure, Axes and a list of plotted elements
        """

        if self.data.points.size == 0:
            raise ValueError(
                "No data points found to evaluate."
            )

        if dim is None:
            dim = (0, 1)
        elif dim[1] < dim[0]:
            dim = dim[::-1]

        if points is None:
            points = (None, None, None)

        # Slicing without copying
        _data = self.data.points[
            slice(*points),
            slice(dim[0], dim[1] + 1, dim[1] - dim[0])
            ]

        # Plot original set or points per cluster?
        if not original:
            if self.labels.size > 0:
                clusterdict = self.labels.clusterdict
                if clusters is None:
                    clusters = list(self.labels.clusterdict.keys())
            else:
                original = True
                clusterdict = None

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

        if plot == "dots":
            plot_props_defaults = {
                'lw': 0,
                'marker': '.',
                'markersize': 4,
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

            plotted = _plots.plot_dots(
                ax=ax, data=_data, original=original,
                clusterdict=clusterdict,
                clusters=clusters,
                dot_props=plot_props_defaults,
                dot_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        elif plot == "scatter":
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

            plotted = _plots.plot_scatter(
                ax=ax, data=_data, original=original,
                clusterdict=clusterdict,
                clusters=clusters,
                scatter_props=plot_props_defaults,
                scatter_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        if plot in ["contour", "contourf", "histogram"]:

            hist_props_defaults = {
                "avoid_zero_count": False,
                "mass": True,
                "mids": True
            }

            if hist_props is not None:
                hist_props_defaults.update(hist_props)

            if plot == "contour":

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

                plotted = _plots.plot_contour(
                    ax=ax, data=_data, original=original,
                    clusterdict=clusterdict,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot == "contourf":
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

                plotted = _plots.plot_contourf(
                    ax=ax, data=_data, original=original,
                    clusterdict=clusterdict,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot == "histogram":
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

                plotted = _plots.plot_histogram(
                    ax=ax, data=_data, original=original,
                    clusterdict=clusterdict,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted

    def pie(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plotted = _plots.pie(self, ax=ax)

        return fig, ax, plotted


class CNNChild(CNN):
    """CNN cluster object subclass.

    Increments the hierarchy level of
    the parent object when instanciated.

    Attributes:
        parent: Reference to parent
    """

    def __init__(self, parent, *args, alias='child', **kwargs):
        super().__init__(*args, alias=alias, **kwargs)
        self.parent = parent
        self.hierarchy_level = parent.hierarchy_level + 1


def TypedDataFrame(columns, dtypes, content=None):
    """Optional constructor to convert CNNRecords to pandas.DataFrame

    """

    assert len(columns) == len(dtypes)

    if content is None:
        content = [[] for i in range(len(columns))]

    df = pd.DataFrame({
        k: pd.array(c, dtype=v)
        for k, v, c in zip(columns, dtypes, content)
        })

    return df


def dist(data: Any):
    """High level wrapper function for :meth:`CNN.dist`.

    A :class:`CNN` instance is created with the given data.

    Args:
        data: Points

    Returns:
        Distance matrix (points x points).
    """

    cobj = CNN(data)
    cobj.calc_dist()

    return cobj.data.distances


class MetaSettings(type):
    """Metaclass to inherit class with class properties

    Classes constructed with this metaclass have a :attr:`__defaults`
    class attribute that can be accessed as a property :attr:`defaults`
    from the class.
    """

    __defaults: Dict[str, str] = {}

    @property
    def defaults(cls):
        return cls.__dict__[f"_{cls.__name__}__defaults"]


class Settings(dict, metaclass=MetaSettings):
    """Class to expose and handle configuration

    Inherits from :py:class:`MetaSettings` to allow access to the class
    attribute :py:attr:`__defaults` as a property :py:attr:`defaults`.

    Also derived from basic type :py:class:`dict`.

    The user can sublclass this class :py:class:`Settings` to provide e.g.
    a different set of default values as :py:attr:`__defaults`.
    """

    # Defaults shared by all instances of this class
    # Name-mangling allows different defaults in subclasses
    __defaults = {
        'default_cnn_cutoff': "1",
        'default_cnn_offset': "0",
        'default_radius_cutoff': "1",
        'default_member_cutoff': "2",
        'float_precision': 'sp',
        'int_precision': 'sp',
        }

    @property
    def defaults(self):
        """Return class attribute from instance"""
        return type(self).__defaults

    __float_precision_map = {
        'hp': np.float16,
        'sp': np.float32,
        'dp': np.float64,
    }

    __int_precision_map = {
        'qp': np.int8,
        'hp': np.int16,
        'sp': np.int32,
        'dp': np.int64,
    }

    @property
    def int_precision_map(cls):
        return cls.__int_precision_map

    @property
    def float_precision_map(cls):
        return cls.__float_precision_map

    @property
    def cfgfile(self):
        return self._cfgfile

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(type(self).__defaults)
        self.update(*args, **kwargs)

        self._cfgfile = None

    def __setitem__(self, key, val):
        if key in type(self).__defaults:
            super().__setitem__(key, val)
        else:
            print(f"Unknown option: {key}")

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def configure(
            self, path: Optional[Union[Path, str]] = None,
            reset: bool = False):
        """Configuration file reading

        Reads a yaml configuration file ``.corerc`` from a given path or
        one of the standard locations in the following order of
        priority:

            - current working directory
            - user home directory

        Args:
            path: Path to a configuration file.
            reset: Reset to defaults. If True, `path` is ignored
               and no configuration file is read.
        """

        if reset:
            self.update(type(self).__defaults)
        else:
            if path is None:
                path = []
            else:
                path = [path]

            path.extend([Path.cwd() / ".corerc",
                         Path.home() / ".corerc"])

            places = iter(path)

            # find configuration file
            while True:
                try:
                    cfgfile = next(places)
                except StopIteration:
                    self._cfgfile = None
                    break
                else:
                    if cfgfile.is_file():
                        with open(cfgfile, 'r') as ymlfile:
                            self.update(yaml.load(
                                ymlfile, Loader=yaml.SafeLoader
                                ))

                        self._cfgfile = cfgfile
                        break


# Configuration setup
settings = Settings()
""":py:class:`Settings`: Module level settings container"""

settings.configure()

if __name__ == "__main__":
    pass
