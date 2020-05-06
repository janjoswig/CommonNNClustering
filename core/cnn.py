#!/usr/bin/env python

"""cnn.py - A Python module for the
   common-nearest-neighbour (CNN) cluster algorithm.

The functionality provided in this module is based on code implemented
by Oliver Lemke in the script collection CNNClustering available on
git-hub (https://github.com/BDGSoftware/CNNClustering.git). Please cite:

    * B. Keller, X. Daura, W. F. van Gunsteren J. Chem. Phys.,
        2010, 132, 074110.
    * O. Lemke, B.G. Keller, J. Chem. Phys., 2016, 145, 164104.
    * O. Lemke, B.G. Keller, Algorithms, 2018, 11, 19.
"""


from collections import Counter, defaultdict, namedtuple, UserList
import functools
from itertools import count
import pickle
from pathlib import Path
import random
import sys
import tempfile
import time
import warnings
from typing import Dict, List, Set, Tuple
from typing import Sequence, Iterable, Iterator, Collection
from typing import Any, Optional, Type, Union, IO

import colorama  # TODO Make this optional or remove completely?
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # TODO make this dependency optional?
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import tqdm
import yaml

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
                    else:
                        print(f"{entry:<10}", end="")
                print("\n" + "-" * 80)

            self.summary.append(record)

        return
    return wrapper


class Labels(np.ndarray):
    """Cluster label assignments"""

    def __new__(cls, sequence: Sequence[int]):
        if sequence is None:
            return None

        obj = np.asarray(sequence, dtype=int).view(cls)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

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

    @functools.cached_property
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

    def sort_by_size(self, member_cutoff=None, max_clusters=None):
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

        noise = 0
        frequencies = Counter(self)
        if 0 in frequencies:
            noise = frequencies.pop(0)

        largest = 0
        order = frequencies.most_common()
        if order:
            largest = order[0][1]

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

        return largest, noise

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

        for add in clusters:
            self[self == add] = 0

        self.sort_by_size()

        return


class Neighbourhoods(UserList):
    """Abstraction class for a neigbourlist

    Stores a neighbourlist in form of a list of *n* sets of *m*
    integers, which are *m* neighbouring point indices for one of the
    *n* points.
    """

    def __init__(self, neighbourhoods, radius):
        super().__init__(neighbourhoods)
        self._radius = radius  # No setter, should always match data

    def __str__(self):
        return f"Neighbourhoods, radius = {self.radius}"

    @property
    def radius(self):
        return self._radius

    @functools.cached_property
    def n_neighbours(self):
        return [len(x) for x in self.neighbourhoods]


class Data:
    """Abstraction class for handling input data

    Bundles points (`Points`), distances (), neighbourhoods () and
    auxillaries like trees.

    """
    def __init__(
            self,
            points=None, dist_matrix=None, map_matrix=None,
            neighbourhoods=None):
        self.points = points
        self.dist_matrix = dist_matrix
        self.map_matrix = map_matrix
        self.neighbourhoods = neighbourhoods

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, x):
        self._points = Points.from_parts(x)
        # TODO This is the right point for choosing the constructor


class Points(np.ndarray):
    """Abstraction class for data points


    """

    def __new__(
            cls,
            p: Optional[np.ndarray] = None,
            edges: Optional[Sequence] = None):
        if p is None:
            p = np.array([])
        obj = p.view(cls)
        obj._edges = edges
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._edges = getattr(obj, "edges", None)

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

    @classmethod
    def from_parts(cls, p: Optional[Sequence]):
        """Alternative constructor

        Use if data is passed as collection of parts, as
            >>> obj = Points.from_parts([[[0],[1]], [[2],[3]]])

        In this way, part edges are taken from the input shape and do
        not have to be specified explicitly.
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

        In this way, part edges are taken from the input shape and do
        not have to be specified explicitly.
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
            edges -- Part edges, marking the end points of the parts
        """

        if data is None:
            return None, None

        data_shape = np.shape(data[0])
        # raises a type error if data is not subscribable

        if np.shape(data_shape)[0] == 0:
            # 1D Sequence passed
            data = [np.array([data])]

        elif np.shape(data_shape)[0] == 1:
            # 2D Sequence of sequences passed
            data = [np.asarray(data)]

        elif np.shape(data_shape)[0] == 2:
            # Sequence of 2D sequences of sequences passed
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


class Summary(list):

    def to_DataFrame(self):
        """Convert list of records to (typed) pandas.DataFrame"""
        pass


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
    """

    def __init__(
            self,
            points: Optional[Any] = None,
            dist_matrix: Optional[Any] = None,
            map_matrix: Optional[Any] = None,
            neighbourhoods: Optional[Any] = None,
            labels: Collection[int] = None,
            alias: str = "root") -> None:

        self.alias = alias        # Descriptive object identifier
        self.hierarchy_level = 0  # See hierarchy_level.setter

        self.data = Data(
            points, dist_matrix, map_matrix, neighbourhoods
            )

        self.labels = labels  # See labels.setter
        self.summary = Summary()
        self._children = None
        self._refindex = None
        self._refindex_rel = None
        self._tree = None
        self._memory_assigned = None
        self._cache = None
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
        and in which format
        """

        self._status = {}

        # Check for data points
        if self.data.points.size > 0:
            self._status["points"] = (True, self.data.points.shape[0])
        else:
            self._status["points"] = (False,)

        if self.data.points.edges is not None:
            self._status["edges"] = (True, len(self.data.points.edges))
        else:
            self._status["edges"] = (False,)

        # Check for point distances
        if self.data.dist_matrix is not None:
            self._status["distances"] = (True, self.data.dist_matrix.shape[0])
        else:
            self._status["distances"] = (False,)

        # Check for neighbourhoods
        if self.data.neighbourhoods is not None:
            self._status["neighbourhoods"] = (True,
                                              len(self.data.neighbourhoods,
                                              self.data.neighbourhoods.radius))
        else:
            self._status["neighbourhoods"] = (False,)

    def __str__(self):

        self.check()
        if self._status["edges"][0]:
            if self._status['edges'][1] > 1:
                if self._status['edges'][1] < 5:
                    edge_str = f"{self._status['edges'][1]}, {self.data.points.edges}"
                else:
                    edge_str = f"{self._status['edges'][1]}, {self.data.points.edges[:5]}"
            else:
                edge_str = f"{self._status['edges'][1]}"
        else:
            edge_str = "None"

        if self._status["points"][0]:
            points_str = f"{self._status['points'][1]}"
            dim_str = f"{self.data.points.shape[1]}"
        else:
            points_str = "None"
            dim_str = f"{self.data.points.shape[1]}"

        if self._status["distances"][0]:
            dist_str = f"{self._status['distances'][1]}"
        else:
            dist_str = "None"

        if self._status["neighbourhoods"][0]:
            neigh_str = f"{self._status['neighbourhoods'][1]}, r={self._status['neighbourhoods'][2]}"
        else:
            neigh_str = "None"

        str_ = (
f"""
===============================================================================
core.cnn.CNN cluster object
-------------------------------------------------------------------------------
alias :                         {self.alias}
hierachy level :                {self.hierarchy_level}

data point shape :              Parts      - {edge_str}
                                Points     - {points_str}
                                Dimensions - {dim_str}

distance matrix calculated :    {dist_str}
neighbour list calculated :     {neigh_str}
clustered :                     {self.check_present(self._labels)}
children :                      {self.check_present(self._children)}
===============================================================================
"""
            )

        return str_

    @timed
    def calc_dist(
            self, v: bool = True, method: str = 'cdist',
            mmap: bool = False,
            mmap_file: Optional[Union[Path, str, IO[bytes]]] = None,
            chunksize: int = 10000, progress: bool = True):
        """Computes a distance matrix (points x points)

        Accesses data points in given data of standard shape

        Args:
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

        progress = not progress

        if method == 'cdist':
            if mmap:
                if mmap_file is None:
                    mmap_file = tempfile.TemporaryFile()

                len_ = self.data.points.shape[0]
                self.data.dist_matrix = np.memmap(
                    mmap_file,
                    dtype=settings.float_precision_map[
                        settings["float_precision"]
                        ],
                    mode='w+',
                    shape=(len_, len_),
                    )
                chunks = np.ceil(len_ / chunksize).astype(int)
                for chunk in tqdm.tqdm(
                        range(chunks), desc="Mapping",
                        disable=progress, unit="Chunks", unit_scale=True,
                        bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (
                            colorama.Style.BRIGHT,
                            colorama.Fore.BLUE,
                            colorama.Fore.RESET
                            )):
                    self.data.dist_matrix[
                            chunk*chunksize: (chunk+1)*chunksize] = cdist(
                        self.data.points[chunk*chunksize: (chunk+1)*chunksize],
                        self.data.points
                        )
            else:
                self.data.dist_matrix = cdist(self.data.points, self.data.points)

        else:
            raise ValueError(
                f"Method {method} not understood."
                "Currently implemented methods:\n"
                "    'cdist'"
                )

    def calc_neighbours_from_dist(self, r):
        """Calculate neighbour list at a given radius

        Requires :attr:`self._dist_matrix`.
        Sets :attr:`self._neighbours`.

        Args:
            r: Radius cutoff

        Returns:
            None
        """

        neighbours = [
            set(np.where((x > 0) & (x < r))[0])
            for x in self.data.dist_matrix
            ]

        self.data.neighbourhoods = Neighbourhoods(neighbours, r)

    @timed
    def map(
            self, method='cdist', mmap=False,
            mmap_file=None, chunksize=10000, progress=True):
        """Computes a map matrix that maps an arbitrary data set to a
        reduced to set"""

        # BROKEN
        if self.__train is None or self.__test is None:
            raise LookupError(
                "Mapping requires a train and a test data set"
                )
        elif self.__train_shape['dimensions'] < self.__test_shape['dimensions']:
            warnings.warn(
                f"Mapping requires the same number of dimension in the train"
                "and the test data set. Reducing test set dimensions to"
                f"{self.__train_shape['dimensions']}.",
                UserWarning
                )
        elif self.__train_shape['dimensions'] > self.__test_shape['dimensions']:
            raise ValueError(
                f"Mapping requires the same number of dimension in the train \
                and the test data set."
                )

        progress = not progress

        if method == 'cdist':
            _train = np.vstack(self.__train) # Data can not be streamed right now
            _test = np.vstack(self.__test)
            if mmap:
                if mmap_file is None:
                    mmap_file = tempfile.TemporaryFile()

                len_train = len(_train)
                len_test = len(_test)
                self.__map_matrix = np.memmap(
                            mmap_file,
                            dtype=settings.float_precision_map[settings["float_precision"]],
                            mode='w+',
                            shape=(len_test, len_train),
                            )
                chunks = np.ceil(len_test / chunksize).astype(int)
                for chunk in tqdm.tqdm(range(chunks), desc="Mapping",
                        disable=progress, unit="Chunks", unit_scale=True,
                        bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (
                            colorama.Style.BRIGHT,
                            colorama.Fore.BLUE,
                            colorama.Fore.RESET)
                            ):
                    self.__map_matrix[chunk*chunksize : (chunk+1)*chunksize] = cdist(
                        _test[chunk*chunksize : (chunk+1)*chunksize], _train
                        )
            else:
                self.__map_matrix = cdist(_test, _train)

        else:
            raise ValueError()

    def dist_hist(  # BROKEN
            self, ax: Optional[Type[mpl.axes.SubplotBase]] = None,
            maxima: bool = False,
            maxima_props: Optional[Dict[str, Any]] = None,
            hist_props: Optional[Dict[str, Any]] = None,
            ax_props: Optional[Dict[str, Any]] = None,
            inter_props: Optional[Dict[str, Any]] = None,
            **kwargs):
        """Make a histogram of distances in the data set

        Args:
            ax: Axes to plot on. If None, Figure and Axes are created.
            maxima: Whether to mark the maxima of the distribution.
                Uses `scipy.signal.argrelextrema`.
            maxima_props: Keyword arguments passed to
                `scipy.signal.argrelextrema` if `maxima` is set
                to True.
            maxima_props: Keyword arguments passed to `numpy.histogram`
                to compute the histogram.
            ax_props: Keyword arguments passed to `ax.set` for styling.
        """

        # TODO Add option for kernel density estimation
        # (scipy.stats.gaussian_kde, statsmodels.nonparametric.kde)

        if self._dist_matrix is None:
            print(
                "Distance matrix not calculated."
                "Calculating distance matrix."
                )
            self.dist(mode=mode, **kwargs)
        _dist_matrix = self._dist_matrix

        # TODO make this a configuration option
        hist_props_defaults = {
            "bins": 100,
            "density": True,
        }

        if hist_props is not None:
            hist_props_defaults.update(hist_props)

        flat_ = np.tril(_dist_matrix).flatten()
        histogram, bins =  np.histogram(
            flat_[flat_ > 0],
            **hist_props_defaults
            )

        binmids = bins[:-1] + (bins[-1] - bins[0]) / ((len(bins) - 1)*2)

        # TODO make this a configuation option
        inter_props_defaults = {
            "ifactor": 0.5,
            "kind": 'linear',
        }

        if inter_props is not None:
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


        ylimit = np.max(histogram)*1.1

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
                                histogram[candidate]+(ylimit/100))
                        )
                    )
        else:
            annotations = None

        ax.set(**ax_props_defaults)

        return fig, ax, line, annotations

    @recorded
    @timed
    def fit(
            self, radius_cutoff: Optional[float] = None,
            cnn_cutoff: Optional[int] = None,
            member_cutoff: Optional[int] = None,
            max_clusters: Optional[int] = None,
            cnn_offset: Optional[int] = None,
            rec: bool = True, v: bool = True,
            method: str = "n",
            policy: str = "progressive",
            ) -> Optional[Tuple[CNNRecord, bool]]:
        """Wraps CNN clustering execution

        This function prepares the clustering and calls an appropriate
        worker function to do a clustering.  How the clustering is done,
        depends on the current data situation and the selected `policy`.
        The clustering can be done either with data points, pre-computed
        pairwise point distances, or pre-computed neighbourhoods as
        input.  Ultimately, neighbourhoods are used during the
        clustering.  Clustering is fast if neighbourhoods are
        pre-computed but this has to be done for each `radius_cutoff`
        separately. Neighbourhoods can be calculated either from data
        points, or pre-computed pairwaise distances.  Storage of
        distances can be costly memory-wise.  If the user chooses
        `policy = "progressive"`, neighbourhoods will be computed from
        either distances (if present) or points before the clustering.
        If the user chooses `policy = "conservative"`, neighbourhoods
        will be computed on-the-fly (online) from either distances (if
        present) or points during the clustering.  This can save memory
        but can be computational more expensive.  Caching can be used
        to achieve the right balance between memory usage and computing
        effort for your situation.

        """

        assert policy in ["progressive", "conservative"]

        # Set params
        params = {  # option name, (user option name, used as type here)
            'radius_cutoff': (radius_cutoff, float),
            'cnn_cutoff': (cnn_cutoff, int),
            'member_cutoff': (member_cutoff, int),
            'cnn_offset': (cnn_offset, int),
            }

        for option, (value, type_) in params.items():
            if value is None:
                default = f"default_{option}"
                params[option] = type_(settings.get(
                    default, settings.defaults.get(default)
                    ))
            else:
                params[option] = params[option][1](params[option][0])

        params["cnn_cutoff"] -= params["cnn_offset"]
        assert params["cnn_cutoff"] >= 0

        # Check data situation
        self.check()

        # Neighbourhoods calculated?
        if (self._status["neighbourhoods"][0]
                and self.data.neighbourhoods.radius == params["radius_cutoff"]):
            # Fit from pre-computed neighbourhoods,
            # no matter what the policy is
            fit_fxn = _fits.fit_from_neighbours
            fit_args = (params["cnn_cutoff"], self.data.neighbourhoods)
            # Fit from List[Set[int]]
            # TODO: Allow different methods and data structures

        # Distances calculated?
        elif self._status["distances"][0]:
            if policy == "progressive":
                # Pre-compute neighbourhoods from distances
                self.calc_neighbours_from_dist(params["radius_cutoff"])
                fit_fxn = _fits.fit_from_neighbours
                fit_args = (params["cnn_cutoff"], self.data.neighbourhoods)

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
                raise NotImplementedError()

        # Call clustering
        self.labels = fit_fxn(*fit_args)

        # TODO: Make this optional?
        # Sort by size and filter
        largest, noise = self.labels.sort_by_size(
            member_cutoff=params["member_cutoff"],
            max_clusters=max_clusters
            )

        if rec:
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

    def cKDtree(self, **kwargs):
        """Wrapper for `scipy.spatial.cKDTree`

        Sets CNN._tree.

        Args:
            **kwargs: Passed to `scipy.spatial.cKDTree`
        """

        self._tree = cKDTree(self._data, **kwargs)

    @staticmethod
    def get_neighbours(
            a: Type[np.ndarray], B: Type[np.ndarray],
            r: float) -> List[int]:
        """Compute neighbours of a point by (squared) distance

        Args:
            a: Point in *n* dimensions; shape (n,)
            B: *m* Points *n* dimensions; shape (m, n)
            r: squared distance cut-off
        Returns:
            Array of indices of points in `B` that are neighbours
            of `a` within `r`."""

        # r = r**2
        return np.where(np.sum((B - a)**2, axis=1) < r)[0]

    @timed
    def predict(  # BROKEN
            self, radius_cutoff: Optional[float] = None,
            cnn_cutoff: Optional[int] = None,
            member_cutoff: Optional[int] = None,
            include_all: bool = True, same_tol=1e-8, memorize: bool = True,
            clusters: Optional[List[int]] = None, purge: bool = False,
            cnn_offset: Optional[int] = None, behaviour="lookup",
            method='plain', progress=True, **kwargs
            ) -> None:
        """
        Predict labels for points in a test set on the basis of assigned
        labels to a train set by :meth:`CNN.fit`

        Parameters
        ----------
        radius_cutoff : float, default=1.0
            Used to find nearest neighbours within distance r

        cnn_cutoff : int, default=1
            Similarity criterion; Points of the same cluster must have
            at least n common nearest neighbours

        member_cutoff : int, default=1
            Clusters must have more than m points or are declared noise

        include_all : bool, default=True
            If False, keep cluster assignment for points in the test set
            that have a maximum distance of `same_tol` to a point
            in the train set, i.e. they are (essentially the same point)
            (currently not implemented)

        same_tol : float, default=1e-8
            Distance cutoff to treat points as the same, if
            `include_all` is False

        clusters : List[int], Optional, default=None
            Predict assignment of points only with respect to this list
            of clusters

        purge : bool, default=False
            If True, reinitalise predicted labels.  Override assignment
            memory.

        memorize : bool, default=True
            If True, remember which points in the test set have been
            already assigned and exclude them from future predictions

        cnn_offset : int, default=0
            Mainly for backwards compatibility; Modifies the the
            cnn_cutoff

        behaviour : str, default="lookup"
            Controlls how the predictor operates:

            * "lookup", Use distance matrices CNN.train_dist_matrix and
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

        if radius_cutoff is None:
            radius_cutoff = float(
                settings.get(
                    'default_radius_cutoff',
                    settings.defaults.get('default_radius_cutoff')
                    )
                )
        if cnn_cutoff is None:
            cnn_cutoff = int(
                settings.get(
                    'default_cnn_cutoff',
                    settings.defaults.get('default_cnn_cutoff')
                    )
                )
        if member_cutoff is None:
            member_cutoff = int(
                settings.get(
                    'default_member_cutoff',
                    settings.defaults.get('default_member_cutoff')
                    )
                )
        if cnn_offset is None:
            cnn_offset = int(
                settings.get(
                    'default_cnn_offset',
                    settings.defaults.get('default_cnn_offset')
                    )
                )

        cnn_cutoff -= cnn_offset

        # TODO: Store a vstacked version in the first place
        _test = np.vstack(self.__test)
        # _map = self.__map_matrix

        len_ = len(_test)

        # TODO: Decouple memorize?
        if purge or (clusters is None):
            self.__test_labels = np.zeros(len_).astype(int)
            self.__memory_assigned = np.ones(len_).astype(bool)
            if clusters is None:
                clusters = list(range(1, len(self.__train_clusterdict)+1))

        else:

            if self.__memory_assigned is None:
                self.__memory_assigned = np.ones(len_).astype(bool)

            if self.__test_labels is None:
                self.__test_labels = np.zeros(len_).astype(int)

            for cluster in clusters:
                self.__memory_assigned[self.__test_labels == cluster] = True
                self.__test_labels[self.__test_labels == cluster] = 0

            _test = _test[self.__memory_assigned]
            # _map = self.__map_matrix[self.__memory_assigned]

        progress = not progress

        if behaviour == "on-the-fly":
            if method == "plain":
                # TODO: Store a vstacked version in the first place
                _train = np.vstack(self.__train)

                r = radius_cutoff**2

                _test_labels = []

                for candidate in tqdm.tqdm(_test, desc="Predicting",
                    disable=progress, unit="Points", unit_scale=True,
                    bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (colorama.Style.BRIGHT, colorama.Fore.BLUE, colorama.Fore.RESET)):
                    _test_labels.append(0)
                    neighbours = self.get_neighbours(
                        candidate, _train, r
                        )

                    # TODO: Decouple this reduction if clusters is None
                    try:
                        neighbours = neighbours[
                            np.isin(self.__train_labels[neighbours], clusters)
                            ]
                    except IndexError:
                        pass
                    else:
                        for neighbour in neighbours:
                            neighbour_neighbours = self.get_neighbours(
                                _train[neighbour], _train, r
                                )

                            # TODO: Decouple this reduction if clusters is None
                            try:
                                neighbour_neighbours = neighbour_neighbours[
                                    np.isin(
                                        self.__train_labels[neighbour_neighbours],
                                        clusters
                                        )
                                    ]
                            except IndexError:
                                pass
                            else:
                                if self.check_similarity_array(
                                    neighbours, neighbour_neighbours, cnn_cutoff
                                    ):
                                    _test_labels[-1] = self.__train_labels[neighbour]
                                    # break after first match
                                    break
            else:
                raise ValueError()

        elif behaviour == "lookup":
            _map = self.__map_matrix[self.__memory_assigned]
            _test_labels = []

            for candidate in tqdm.tqdm(range(len(_test)), desc="Predicting",
                disable=progress, unit="Points", unit_scale=True,
                bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (colorama.Style.BRIGHT, colorama.Fore.BLUE, colorama.Fore.RESET)):

                _test_labels.append(0)
                neighbours = np.where(
                    _map[candidate] < radius_cutoff
                    )[0]

                # TODO: Decouple this reduction if clusters is None
                try:
                    neighbours = neighbours[
                        np.isin(self.__train_labels[neighbours], clusters)
                        ]
                except IndexError:
                    pass
                else:
                    for neighbour in neighbours:
                        neighbour_neighbours = np.where(
                        (self.__train_dist_matrix[neighbour] < radius_cutoff) &
                        (self.__train_dist_matrix[neighbour] > 0)
                        )[0]

                        try:
                            # TODO: Decouple this reduction if clusters is None
                            neighbour_neighbours = neighbour_neighbours[
                                np.isin(
                                    self.__train_labels[neighbour_neighbours],
                                    clusters
                                    )
                                ]
                        except IndexError:
                            pass
                        else:
                            if self.check_similarity_array(
                                neighbours, neighbour_neighbours, cnn_cutoff
                                ):
                                _test_labels[-1] = self.__train_labels[neighbour]
                                # break after first match

                                break

        elif behaviour == "tree":
            if self.__train_tree is None:
                raise LookupError(
"No search tree build for train data. Use CNN.kdtree(mode='train, **kwargs) first."
                    )

            _train = np.vstack(self.__train)

            _test_labels = []

            for candidate in tqdm.tqdm(_test, desc="Predicting",
                disable=progress, unit="Points", unit_scale=True,
                bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (colorama.Style.BRIGHT, colorama.Fore.BLUE, colorama.Fore.RESET)):
                _test_labels.append(0)
                neighbours = np.asarray(self.__train_tree.query_ball_point(
                    candidate, radius_cutoff, **kwargs
                    ))

                # TODO: Decouple this reduction if clusters is None
                try:
                    neighbours = neighbours[
                        np.isin(self.__train_labels[neighbours], clusters)
                        ]
                except IndexError:
                    pass
                else:
                    for neighbour in neighbours:
                        neighbour_neighbours = np.asarray(self.__train_tree.query_ball_point(
                            _train[neighbour], radius_cutoff, **kwargs
                            ))
                        try:
                            # TODO: Decouple this reduction if clusters is None
                            neighbour_neighbours = neighbour_neighbours[
                                np.isin(
                                    self.__train_labels[neighbour_neighbours],
                                    clusters
                                    )
                                ]
                        except IndexError:
                            pass
                        else:
                            if self.check_similarity_array(
                                neighbours, neighbour_neighbours, cnn_cutoff
                                ):
                                _test_labels[-1] = self.__train_labels[neighbour]
                                # break after first match
                                break
        else:
            raise ValueError(
f'Behaviour "{behaviour}" not known. Must be one of "on-the-fly", "lookup" or "tree"'
            )

        self.__test_labels[self.__memory_assigned] = _test_labels
        self.labels2dict(mode="test")

        if memorize:
            self.__memory_assigned[np.where(self.__test_labels > 0)[0]] = False

    @staticmethod
    def check_similarity_sequence(a: Sequence[int], b: Sequence[int], c: int) -> bool:
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

    def isolate(self, purge=True):
        """Isolates points per clusters based on a cluster result"""

        if purge or self._children is None:
            self._children = defaultdict(lambda: CNNChild(self))

        for key, _cluster in  self._clusterdict.items():
            # TODO: What if no noise?
            if len(_cluster) > 0:
                _cluster = np.asarray(_cluster)
                ref_index = []
                ref_index_rel = []
                cluster_data = []
                part_startpoint = 0

                if self._refindex is None:
                    ref_index.extend(_cluster)
                    ref_index_rel = ref_index
                else:
                    ref_index.extend(self._refindex[_cluster])
                    ref_index_rel.extend(_cluster)

                for part in range(self._shape['parts']):
                    part_endpoint = part_startpoint \
                        + self._shape['points'][part] -1

                    cluster_data.append(
                        self._data[part][_cluster[
                            np.where(
                                (_cluster
                                >= part_startpoint)
                                &
                                (_cluster
                                <= part_endpoint))[0]] - part_startpoint]
                            )
                    part_startpoint = np.copy(part_endpoint)
                    part_startpoint += 1

                self._children[key].alias = f'child No. {key}'
                self._children[key].data = cluster_data
                self._children[key]._refindex = np.asarray(ref_index)
                self._children[key]._refindex_rel = np.asarray(ref_index_rel)
        return

    def reel(self, deep: int = 1):
        """Wrap up assigments of lower hierarchy levels

        Args:
            deep: How many lower levels to consider.
        """

        if self._children is None:
            raise LookupError(
                "No child clusters isolated"
                             )
        # TODO: Implement "deep" for degree of decent into hierarchy structure

        for _cluster in self._children.values():
            n_clusters = max(self._clusterdict)
            if _cluster._labels is not None:
                if self.hierarchy_level == 0:
                    self._labels[
                        _cluster._refindex[
                        np.where(_cluster._labels == 0)[0]
                        ]
                    ] = 0
                else:
                    self._labels[
                        _cluster._refindex_rel[
                        np.where(_cluster._labels == 0)[0]
                        ]
                    ] = 0

                for _label in _cluster._labels[_cluster._labels > 1]:
                    if self.hierarchy_level == 0:
                        self._labels[
                            _cluster._refindex[
                            np.where(_cluster._labels == _label)[0]
                            ]
                        ] = _label + n_clusters
                    else:
                        self._labels[
                            _cluster._refindex_rel[
                            np.where(_cluster._labels == _label)[0]
                            ]
                        ] = _label + n_clusters

            self.clean()
            self.labels2dict()

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
            if self.labels is not None:
                if clusters is None:
                    clusters = list(self.labels.clusterdict.keys())
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
                clusterdict=self.labels.clusterdict,
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
                clusterdict=self.labels.clusterdict,
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

                plotted = _plots.plot_contour(
                    ax=ax, data=_data, original=original,
                    clusterdict=self.labels.clusterdict,
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

                plotted = _plots.plot_contourf(
                    ax=ax, data=_data, original=original,
                    clusterdict=self.labels.clusterdict,
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

                plotted = _plots.plot_histogram(
                    ax=ax, data=_data, original=original,
                    clusterdict=self.labels.clusterdict,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted


class CNNChild(CNN):
    """CNN cluster object subclass. Increments the hierarchy level of
    the parent object when instanciated."""

    def __init__(self, parent, alias='child'):
        super().__init__()
        self.hierarchy_level = parent.hierarchy_level +1
        self.alias = alias


def TypedDataFrame(columns, dtypes, content=None, index=None):
    """Obsolete: eliminate pandas dependency

    summary = TypedDataFrame(
            columns=self.Record._fields,
            dtypes=self._record_dtypes
            )

    self._record_dtypes = [
        pd.Int64Dtype(), np.float64, pd.Int64Dtype(), pd.Int64Dtype(),
        pd.Int64Dtype(), pd.Int64Dtype(), np.float64, np.float64,
        np.float64
        ]
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

    cobj = CNN(data=data)
    cobj.dist()

    return cobj._dist_matrix


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

    Inherits from :class:`MetaSettings` to allow access to the class
    attribute :attr:`__defaults` as a property :attr:`defaults`.

    Also derived from basic type :class:`dict`.

    The user can sublclass this class :class:`Settings` to provide e.g.
    a different set of default values as :attr:`__defaults`.
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
""":obj:`Settings`: Module level settings container"""

settings.configure()

if __name__ == "__main__":
    pass
