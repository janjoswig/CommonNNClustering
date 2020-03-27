#!/usr/bin/env python

"""cnn.py - A Python module for the
   common-nearest-neighbour (CNN) cluster algorithm.

The functionality provided in this module is based on code implemented
by Oliver Lemke in the script collection CNNClustering available on
git-hub (https://github.com/BDGSoftware/CNNClustering.git). Please cite:

    * B. Keller, X. Daura, W. F. van Gunsteren J. Chem. Phys., 2010, 132, 074110.
    * O. Lemke, B.G. Keller, J. Chem. Phys., 2016, 145, 164104.
    * O. Lemke, B.G. Keller, Algorithms, 2018, 11, 19.
"""


from collections import defaultdict, namedtuple
import copy
from functools import wraps
import pickle
from pathlib import Path
import random
import tempfile
import time
import warnings
from typing import Dict, List, Sequence, Tuple
from typing import Any, Iterable, Iterator, Optional, Type, Union

import colorama
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # TODO make this dependency optional?
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sortedcontainers import SortedList
import tqdm
import yaml


def timed(function_):
    """Decorator to measure execution time.

    Forwards the output of the wrapped function and measured execution
    time.
    """

    @wraps(function_)
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
            return wrapped, stopped
    return wrapper


def recorded(function_):
    """Decorator to format function feedback.

    Feedback needs to be pandas series in record format.  If execution
    time was measured, this will be included in the summary.
    """

    @wraps(function_)
    def wrapper(self, *args, **kwargs):
        wrapped = function_(self, *args, **kwargs)
        if wrapped is not None:
            print(f'recording: ...')
            if len(wrapped) > 1:
                wrapped[-2]['time'] = wrapped[-1]
                # print(f'recording: ... \n{wrapped[-2]}')
                self.summary = self.summary.append(
                    wrapped[-2], ignore_index=True
                    )
            else:
                self.summary = self.summary.append(
                    wrapped, ignore_index=True
                    )
        return
    return wrapper


class CNN:
    """CNN cluster object class"""

    @staticmethod
    def get_shape(data: Any):
        """Maintain data in universal shape

        Analyses the format of given data and fits it into standard
        format (parts, points, dimensions).

        Args:
            data: Either None or
                * a 1D sequence of length x,
                    interpreted as 1 point in x dimension
                * a 2D sequence of length x (rows) times y (columns),
                    interpreted as x points in y dimension
                * a list of 2D sequences,
                    interpreted as groups of points

        Returns: a numpy.ndarray of shape (parts, points, dimension)
        """

        if data is None:
            return None, {
                'parts': None,
                'points': None,
                'dimensions': None,
                }
        else:
            data_shape = np.shape(data[0])
            # raises a type error if data is not subscriptable
            if np.shape(data_shape)[0] == 0:
                # 1D Sequence passed
                data = np.array([[data]])

            elif np.shape(data_shape)[0] == 1:
                # 2D Sequence of sequences passed
                data = np.array([data])

            elif np.shape(data_shape)[0] == 2:
                # List of 2D sequences of sequences passed
                data = np.array([np.asarray(x) for x in data])

            else:
                raise ValueError(
                    f"Data shape {data_shape} not allowed"
                    )

            return data, {
                'parts': np.shape(data)[0],
                'points': [np.shape(x)[0] for x in data],
                'dimensions': np.unique([np.shape(x)[1] for x in data])[0],
                }

    def __init__(
            self, data: Optional[Any] = None, alias: str = 'root',
            dist_matrix: Optional[Any] = None,
            map_matrix: Optional[Any] = None) -> None:

        self.alias = alias  # Descriptive object identifier
        self._hierarchy_level = 0

        # generic function feedback data container for CNN.cluster()
        self.record = namedtuple(
            'ClusterRecord', [
                settings.get('record_points',
                             settings.defaults['record_points']),
                settings.get('record_radius_cutoff',
                             settings.defaults['record_radius_cutoff']),
                settings.get('record_cnn_cutoff',
                             settings.defaults['record_cnn_cutoff']),
                settings.get('record_member_cutoff',
                             settings.defaults['record_member_cutoff']),
                settings.get('record_max_cluster',
                             settings.defaults['record_max_cluster']),
                settings.get('record_n_cluster',
                             settings.defaults['record_n_cluster']),
                settings.get('record_largest',
                             settings.defaults['record_largest']),
                settings.get('record_noise',
                             settings.defaults['record_noise']),
                settings.get('record_time',
                             settings.defaults['record_time']),
                ]
            )

        self._record_dtypes = [
            pd.Int64Dtype(), np.float64, pd.Int64Dtype(), pd.Int64Dtype(),
            pd.Int64Dtype(), pd.Int64Dtype(), np.float64, np.float64,
            np.float64
            ]

        self._data = data
        self._dist_matrix = dist_matrix
        self._map_matrix = map_matrix
        self._clusterdict = None
        self._labels = None
        self.summary = TypedDataFrame(
            columns=self.record._fields,
            dtypes=self._record_dtypes
            )
        self._children = None
        self._refindex = None
        self._refindex_rel = None
        self._tree = None
        self._memory_assigned = None
        self._cache = None

    @property
    def hierarchy_level(self):
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, level):
        self._hierarchy_level = int(level)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x):
        # TODO control string, array, hdf5 file object handling
        self._data, self._shape = self.get_shape(x)


    @property
    def dist_matrix(self):
        return self._dist_matrix

    @dist_matrix.setter
    def dist_matrix(self, x):
        # TODO control string, array, hdf5 file object handling
        self._dist_matrix = x

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, s):
        self._shape = s

    @property
    def clusterdict(self):
        return self._clusterdict

    @clusterdict.setter
    def clusterdict(self, d):
        self._clusterdict = d

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, l):
        self._labels = l

    @property
    def map_matrix(self):
        return self.__map_matrix

    @property
    def tree(self):
        return self._tree

    @property
    def memory_assigned(self):
        return self._memory_assigned

    @memory_assigned.setter
    def memory_assigned(self, mem):
        self._memory_assigned = mem

    @property
    def children(self):
        return self._children

    @property
    def refindex(self):
        return self._refindex

    @property
    def refindex_rel(self):
        return self._refindex_rel

    def check(self):
        """Check the state of the :class:`CNN` instance object"""

        if self._data is not None:
            self.data_present = True
            self.shape_str = {**self.shape}
            self.shape_str['points'] = (
                sum(self._shape_str['points']),
                self._shape_str['points'][:5]
                )
            if len(self._shape['points']) > 5:
                self._shape_str['points'] += ["..."]
        else:
            self.data_present = False
            self._shape_str = {"parts": None, "points": None, "dimensions": None}

        if self._dist_matrix is not None:
            self.dist_matrix_present = True
        else:
            self.dist_matrix_present = False

        if self._clusterdict is not None:
            self.clusters_present = True
        else:
            self.clusters_present = False

        if self._children is not None:
            self.children_present = True
        else:
            self.children_present = False

    def __str__(self):
        self.check()

        return f"""{colorama.Fore.BLUE}cnn.CNN cluster object{colorama.Fore.RESET}
------------------------------------------------------------------------------------
alias :                         {self.alias}
hierachy level :                {self.hierarchy_level}

data shape :                    Parts      - {self._shape_str["parts"]}
                                Points     - {self._shape_str["points"]}
                                Dimensions - {self._shape_str["dimensions"]}

distance matrix calculated :    {self.dist_matrix_present}
clustered :                     {self.clusters_present}
children :                      {self.children_present}
------------------------------------------------------------------------------------
"""

    def load(self, f: Union[Path, str], **kwargs) -> None:
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

        Keyword Args:
            **kwargs: Passed to loader.
        """
        # add load option for dist_matrix, map_matrix

        extension = Path(f).suffix

        case_ = {
            '.p' : lambda: pickle.load(
                open(f, 'rb'),
                **kwargs
                ),
            '.npy': lambda: np.load(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             '': lambda: np.loadtxt(
                 f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             '.xvg': lambda: np.loadtxt(
                 f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
            '.dat': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             }
        data = case_.get(
            extension,
            lambda: print(f"Unknown filename extension {extension}")
            )()

        self._data, self._shape = self.get_shape(data)

    def delete(self):
        """Clear :attr:`data` and :attr:`shape`"""

        del self._data
        del self._shape
        self._data = None
        self._shape = None

    def save(self, file_, content, **kwargs):
        """Saves content to file"""

        extension = file_.rsplit('.', 1)[-1]
        if len(extension) == 1:
            extension = ''
        {
            'p' : lambda: pickle.dump(open(file_, 'wb'), content),
            'npy': lambda: np.save(file_, content, **kwargs),
            '': lambda: np.savetxt(file_, content, **kwargs),
        }.get(extension,
              f"Unknown filename extension .{extension}")()

    def cut(
            self, parts=(None, None, None), points=(None, None, None),
            dimensions=(None, None, None)):

        """Reduce the data set.

        For each data set level (parts, points, dimensions),
        a tuple (start:stop:step) can be specified. The corresponding
        level is cut using :func:`slice`.
        """

        self._data = [
            x[slice(*points), slice(*dimensions)]
            for x in self.__test[slice(*parts)]
            ]

        self._data, self._shape = self.get_shape(self._data)

    def loop_over_points(self) -> Iterator:
        """Iterate over all points of all parts

        Returns:
            Iterator over points
        """

        if self._data:
            for i in self._data:
                for j in i:
                    yield j
        else:
            yield from ()

    @timed
    def dist(
            self, v: bool = True, method: str = 'cdist',
            mmap: bool = False, mmap_file: Optional[str] = None,
            chunksize: int = 10000, progress: bool = True):
        """Computes a distance matrix (points x points)

        Accesses data points in given data of standard shape
        (parts, points, dimensions)
        Args:
            v: Be chatty

        """

        if not self._data:
            return

        progress = not progress

        if method == 'cdist':
            points = np.vstack(self._data)
            # Data can not be streamed right now

            if mmap:
                if mmap_file is None:
                    mmap_file = tempfile.TemporaryFile()

                len_ = len(points)
                _distance_matrix = np.memmap(
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
                    _distance_matrix[chunk*chunksize: (chunk+1)*chunksize] = cdist(
                        points[chunk*chunksize: (chunk+1)*chunksize], points
                        )
            else:
                _distance_matrix = cdist(points, points)

        else:
            raise ValueError(
                f"Method {method} not understood."
                "Currently implemented methods:\n"
                "    'cdist'"
                )

            self._dist_matrix = _distance_matrix

    @timed
    def map(  # BROKEN
            self, method='cdist', mmap=False,
            mmap_file=None, chunksize=10000, progress=True): # nearest=None):
        """Computes a map matrix that maps an arbitrary data set to a
        reduced to set"""

        if self.__train is None or self.__test is None:
            raise LookupError(
                "Mapping requires a train and a test data set"
                )
        elif self.__train_shape['dimensions'] < self.__test_shape['dimensions']:
            warnings.warn(
                f"Mapping requires the same number of dimension in the train \
                  and the test data set. Reducing test set dimensions to \
                  {self.__train_shape['dimensions']}.",
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

        # TODO make this a configuation option
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
                np.ceil(len(binmids)*ifactor)
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
            member_cutoff: int = None,
            max_clusters: Optional[int] = None,
            cnn_offset: int = None,
            rec: bool = True, v=True
            ) -> Optional[pd.DataFrame]:
        """Executes the CNN clustering

        """
        # go = time.time()
        # print("Function called")

        # TODO: Refactor DRY
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
        assert cnn_cutoff >= 0

        if self._dist_matrix is None:
            self.dist()

        # print(f"Data checked: {time.time() - go}")

        n_points = len(self._dist_matrix)

        # calculate neighbour list
        neighbours = np.asarray([
            np.where((x > 0) & (x < radius_cutoff))[0]
            for x in self._dist_matrix
            ])

        n_neighbours = np.asarray([len(x) for x in neighbours])
        include = np.ones(len(neighbours), dtype=bool)
        include[np.where(n_neighbours < cnn_cutoff)[0]] = False
        # include[np.where(n_neighbours <= cnn_cutoff+1)[0]] = False

        _clusterdict = defaultdict(SortedList)
        _clusterdict[0].update(np.where(include == False)[0])
        _labels = np.zeros(n_points).astype(int)
        current = 1

        # print(f"Initialisation done: {time.time() - go}")

        enough = False
        while any(include) and not enough:
            # find point with currently highest neighbour count
            point = np.where(
                (n_neighbours == np.max(n_neighbours[include]))
                & (include == True)
                )[0][0]
            # point = np.argmax(n_neighbours[include])

            _clusterdict[current].add(point)
            new_point_added = True
            _labels[point] = current
            include[point] = False
            # print(f"Opened cluster {current}: {time.time() - go}")
            # done = 0
            while new_point_added:
                new_point_added = False
                # for member in _clusterdict[current][done:]:
                for member in [
                    added_point for added_point in _clusterdict[current]
                    if any(include[neighbours[added_point]])
                    ]:
                    # Is the SortedList dangerous here?
                    for neighbour in neighbours[member][include[neighbours[member]]]:
                        common_neighbours = (
                            set(neighbours[member])
                            & set(neighbours[neighbour])
                            )

                        if len(common_neighbours) >= cnn_cutoff:
                        # and (point in neighbours[neighbour])
                        # and (neighbour in neighbours[point]):
                            _clusterdict[current].add(neighbour)
                            new_point_added = True
                            _labels[neighbour] = current
                            include[neighbour] = False

                # done += 1
            current += 1

            if max_clusters is not None:
                if current == max_clusters+1:
                    enough = True

        # print(f"Clustering done: {time.time() - go}")

        clusters_no_noise = {
            y: _clusterdict[y]
            for y in _clusterdict if y != 0
            }

        # print(f"Make clusters_no_noise copy: {time.time() - go}")

        too_small = [
            _clusterdict.pop(y)
            for y in [x[0]
            for x in clusters_no_noise.items() if len(x[1]) <= member_cutoff]
            ]

        if len(too_small) > 0:
            for entry in too_small:
                _clusterdict[0].update(entry)

        for x in set(_labels):
            if x not in set(_clusterdict):
                _labels[_labels == x] = 0

        # print(f"Declared small clusters as noise: {time.time() - go}")

        if len(clusters_no_noise) == 0:
            largest = 0
        else:
            largest = len(_clusterdict[1 + np.argmax([
                len(x)
                for x in clusters_no_noise.values()
                    ])]) / n_points

        # print(f"Found largest cluster: {time.time() - go}")

        self._clusterdict = _clusterdict
        self._labels = _labels
        self.clean()
        self.labels2dict()

        # print(f"Updated state: {time.time() - go}")
        cresult = TypedDataFrame(
            self.record._fields,
            self.__record_dtypes,
            content=[
                [n_points],
                [radius_cutoff],
                [cnn_cutoff],
                [member_cutoff],
                [max_clusters],
                [len(self._clusterdict) -1],
                [largest],
                [len(self._clusterdict[0]) / n_points],
                [None],
                ],
            )

        if v:
            print("\n" + "-"*72)
            print(
                cresult[list(self.record._fields)[:-1]].to_string(
                    na_rep="None", index=False, line_width=80,
                    header=[
                        "  #points  ", "  R  ", "  N  ", "  M  ",
                        "  max  ", "  #clusters  ", "  %largest  ",
                        "  %noise  "
                        ],
                    justify="center"
                    ))
            print("-"*72)

        if rec:
            return(cresult)
        return

    def merge(self, clusters, which='labels'):
        """Merge a list of clusters into one"""

        if len(clusters) < 2:
            raise ValueError(
                "List of cluster needs to habe at least 2 elements"
                )

        if not isinstance(clusters, list):
            clusters = list(clusters)
        clusters.sort()

        base = clusters[0]

        if which == "labels":
            _labels = self._labels

            for add in clusters[1:]:
                _labels[_labels == add] = base

            self._labels = _labels

            self.clean()
            self.labels2dict()

        elif which == "dict":
            raise NotImplementedError()

            dict_ = self._clusterdict

            for add in clusters[1:]:
                dict_[base].update(dict_[add])
                del dict_[add]

            self.clean()
            self.dict2labels()

        else:
            raise ValueError()

        return

    def trash(self, clusters, which='labels'):
        """Merge a list of clusters into noise"""

        if which == "labels":
            _labels = self._labels

            for add in clusters:
                _labels[_labels == add] = 0

            self._labels = _labels

            self.clean()
            self.labels2dict()

        elif which == "dict":
            raise NotImplementedError()

            dict_ = self._clusterdict

            for cluster in clusters:
                dict_[0].update(dict_[cluster])
                del dict_[cluster]

            self.clean()
            self.dict2labels()

        else:
            raise ValueError()

    def kdtree(self, **kwargs):
        """Wrapper for `scipy.spatial.cKDTree`

        Sets CNN._tree.

        Args:
            **kwargs: Passed to `scipy.spatial.cKDTree`
        """

        self._tree = cKDTree(np.vstack(self._data), **kwargs)

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
    def check_similarity(a: Sequence[int], b: Sequence[int], c: int) -> bool:
        """Check if similarity criterion is fulfilled.

        Args:
            a: Sequence of point indices
            b: Sequence of point indices
            c: Similarity cut-off

        Returns:
            True if list `a` and list `b` have at least `c` common
            elements
        """

        if len(set(a).intersection(b)) >= c:
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

    def query_data(self):  # BROKEN
        """Helper function to evaluate user input.

        If data is required sort out in which form it is present and
        return the needed shape.
        """

        return

    def evaluate(  # BROKEN
        self,
        ax: Optional[Type[mpl.axes.SubplotBase]] = None,
        clusters: Optional[List[int]]=None,
        original: bool=False, plot: str='dots',
        parts: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None,
        points: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None,
        dim: Optional[Tuple[int, int]]=None,
        ax_props: Optional[Dict]=None, annotate: bool=True,
        annotate_pos: str="mean", annotate_props: Optional[Dict]=None,
        scatter_props: Optional[Dict]=None,
        scatter_noise_props: Optional[Dict]=None,
        dot_props: Optional[Dict]=None,
        dot_noise_props: Optional[Dict]=None,
        hist_props: Optional[Dict]=None,
        contour_props: Optional[Dict]=None,
        free_energy: bool=True, mask = None,
        threshold=None,
        ):

        """Returns a 2D plot of an original data set or a cluster result

        Args:
            ax: matplotlib.axes._subplots.AxesSubplot, default=None
                The axes to which to add the plot.  If None, a new figure
                with axes will be created.

            clusters : List[int], default=None
                Cluster numbers to include in the plot.  If None, consider
                all.

            original: bool, default=False
                Allows to plot the original data instead of a cluster
                result.  Overrides `clusters`.  Will be considered
                True, if no cluster result is present.

            plot: str, default="dots"
                The kind of plotting method to use.

                * "dots", Use :func:`ax.plot()`

                * "",

            parts: Tuple[int, int, int] (length 3), default=(None, None, None)
                Use a slice (start, stop, stride) on the data parts before
                plotting.

            points: Tuple[int, int, int], default=(None, None, None)
                Use a slice (start, stop, stride) on the data points before
                plotting.

            dim: Tuple[int, int], default=None
                Use these two dimensions for plotting.  If None, uses
                (0, 1).

            annotate: bool, default=True
                If there is a cluster result, plot the cluster numbers.  Uses
                `annotate_pos` to determinte the position of the
                annotations.

            annotate_pos: str or List[Tuple[int, int]], default="mean"
                Where to put the cluster number annotation.  Can be one of:

                * "mean", Use the cluster mean

                * "random", Use a random point of the cluster

                Alternatively a list of x, y positions can be passed to set
                a specific point for each cluster (Not yet implemented)

            annotate_props: Dict, default=None
                Dictionary of keyword arguments passed to
                :func:`ax.annotate(**kwargs)`.

            ax_props: Dict, default=None
                Dictionary of `ax` properties to apply after
                plotting via :func:`ax.set(**ax_props)`.  If None, uses
                defaults that can be also defined in the configuration file.

            (hist, contour, dot, scatter, dot_noise, scatter_noise)_props: Dict, default=None
                Dictionaries of keyword arguments passed to various
                functions.  If None, uses
                defaults that can be also defined in the configuration file.

            mask: Sequence[bool]

        Returns:
            List of plotted elements
        """

        _data, _ = self.query_data(mode=mode)
        if dim is None:
            dim = (0, 1)
        elif dim[1] < dim[0]:
            dim = dim[::-1]

        if parts is None:
            parts = (None, None, None)

        if points is None:
            points = (None, None, None)

        _data = [
            x[slice(*points), slice(dim[0], dim[1]+1, dim[1]-dim[0])]
            for x in _data[slice(*parts)]
            ]

        _data = np.vstack(_data)
        if mask is not None:
            _data = _data[np.asarray(mask)]

        try:
            items = self._clusterdict.items()
            if clusters is None:
                clusters = list(range(len( items )))
        except AttributeError:
            original = True

        # TODO make this a configuation option
        ax_props_defaults = {
            "xlabel": "$x$",
            "ylabel": "$y$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        annotate_props_defaults = {
            }

        if annotate_props is not None:
            annotate_props_defaults.update(annotate_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # List of axes objects to return for faster access
        plotted = []

        if plot == 'dots':
            # TODO make this a configuation option
            dot_props_defaults = {
                'lw': 0,
                'marker': '.',
                'markersize': 4,
                'markeredgecolor': 'none',
                }

            if dot_props is not None:
                dot_props_defaults.update(dot_props)

            dot_noise_props_defaults = {
                'color': 'none',
                'lw': 0,
                'marker': '.',
                'markersize': 4,
                'markerfacecolor': 'k',
                'markeredgecolor': 'none',
                'alpha': 0.3
                }

            if dot_noise_props is not None:
                dot_noise_props_defaults.update(dot_noise_props)

            if original:
                # Plot the original data
                plotted.append(ax.plot(
                    _data[:, 0],
                    _data[:, 1],
                    **dot_props_defaults
                    ))

            else:
                # Loop through the cluster result
                for cluster, cpoints in items:
                    # plot if cluster is in the list of considered clusters
                    if cluster in clusters:

                        # treat noise differently
                        if cluster == 0:
                            plotted.append(ax.plot(
                                _data[cpoints, 0],
                                _data[cpoints, 1],
                                **dot_noise_props_defaults
                                ))

                        else:
                            plotted.append(ax.plot(
                                _data[cpoints, 0],
                                _data[cpoints, 1],
                                **dot_props_defaults
                                ))

                            if annotate:
                                if annotate_pos == "mean":
                                    xpos = np.mean(_data[cpoints, 0])
                                    ypos = np.mean(_data[cpoints, 1])

                                elif annotate_pos == "random":
                                    choosen = random.sample(
                                        cpoints, 1
                                        )
                                    xpos = _data[choosen, 0]
                                    ypos = _data[choosen, 1]

                                else:
                                    raise ValueError()

                                plotted.append(ax.annotate(
                                    f"{cluster}",
                                    xy=(xpos, ypos),
                                    **annotate_props_defaults
                                    ))

        elif plot == 'scatter':

            scatter_props_defaults = {
                's': 10,
            }

            if scatter_props is not None:
                scatter_props_defaults.update(scatter_props)

            scatter_noise_props_defaults = {
                'color': 'k',
                's': 10,
                'alpha': 0.5
            }

            if scatter_noise_props is not None:
                scatter_noise_props_defaults.update(scatter_noise_props)

            if original:
                plotted.append(ax.scatter(
                    _data[:, 0],
                    _data[:, 1],
                    **scatter_props_defaults
                    ))

            else:
                for cluster, cpoints in items:
                    if cluster in clusters:

                        # treat noise differently
                        if cluster == 0:
                            plotted.append(ax.scatter(
                                _data[cpoints, 0],
                                _data[cpoints, 1],
                                **scatter_noise_props_defaults
                            ))

                        else:
                            plotted.append(ax.scatter(
                                _data[cpoints, 0],
                                _data[cpoints, 1],
                                **scatter_props_defaults
                                ))

                            if annotate:
                                if annotate_pos == "mean":
                                    xpos = np.mean(_data[cpoints, 0])
                                    ypos = np.mean(_data[cpoints, 1])

                                elif annotate_pos == "random":
                                    choosen = random.sample(
                                        cpoints, 1
                                        )
                                    xpos = _data[choosen, 0]
                                    ypos = _data[choosen, 1]

                                else:
                                    raise ValueError()

                                plotted.append(ax.annotate(
                                    f"{cluster}",
                                    xy=(xpos, ypos),
                                    **annotate_props_defaults
                                    ))

        elif plot in ['contour', 'contourf', 'histogram']:

            contour_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

            if contour_props is not None:
                contour_props_defaults.update(contour_props)

            hist_props_defaults = {
                "avoid_zero_count": False,
                "mass": True,
                "mids": True
            }

            if hist_props is not None:
                hist_props_defaults.update(hist_props)

            avoid_zero_count = hist_props_defaults['avoid_zero_count']
            del hist_props_defaults['avoid_zero_count']

            mass = hist_props_defaults['mass']
            del hist_props_defaults['mass']

            mids = hist_props_defaults['mids']
            del hist_props_defaults['mids']

            if original:
                x_, y_, H = get_histogram(
                    _data[:, 0], _data[:, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props_defaults
                )

                if free_energy:
                    dG = np.inf * np.ones(shape=H.shape)

                    nonzero = H.nonzero()
                    dG[nonzero] = -np.log(H[nonzero])
                    dG[nonzero] -= np.min(dG[nonzero])
                    H = dG

                if plot == "histogram":
                    # Plotting the histogram directly
                    # imshow, pcolormesh, NonUniformImage ...
                    # Not implemented, instead return the histogram
                    warnings.warn(
"""Plotting a histogram of the data directly is currently not supported.
Returning the edges and the histogram instead.
""",
UserWarning
                    )
                    return x_, y_, H

                elif plot == 'contour':
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(
                        ax.contour(X, Y, H, **contour_props_defaults)
                        )

                elif plot == 'contourf':
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(
                        ax.contourf(X, Y, H, **contour_props_defaults)
                        )

                else:
                    raise ValueError(
        f"""Plot type {plot} not understood.
        Must be one of 'dots, 'scatter' or 'contour(f)'
        """
                    )

            else:
                for cluster, cpoints in items:
                    if cluster in clusters:
                        x_, y_, H = get_histogram(
                            _data[cpoints, 0], _data[cpoints, 1],
                            mids=mids,
                            mass=mass,
                            avoid_zero_count=avoid_zero_count,
                            hist_props=hist_props_defaults
                        )

                        if free_energy:
                            dG = np.inf * np.ones(shape=H.shape)

                            nonzero = H.nonzero()
                            dG[nonzero] = -np.log(H[nonzero])
                            dG[nonzero] -= np.min(dG[nonzero])
                            H = dG

                        if plot == "histogram":
                            # Plotting the histogram directly
                            # imshow, pcolormesh, NonUniformImage ...
                            # Not implemented, instead return the histogram
                            warnings.warn(
        """Plotting a histogram of the data directly is currently not supported.
        Returning the edges and the histogram instead.
        """,
        UserWarning
                            )
                            return x_, y_, H

                        elif plot == 'contour':
                            X, Y = np.meshgrid(x_, y_)
                            plotted.append(
                                ax.contour(X, Y, H, **contour_props_defaults)
                                )

                        elif plot == 'contourf':
                            X, Y = np.meshgrid(x_, y_)
                            plotted.append(
                                ax.contourf(X, Y, H, **contour_props_defaults)
                                )

                        else:
                            raise ValueError(
                f"""Plot type {plot} not understood.
                Must be one of 'dots, 'scatter' or 'contour(f)'
                """
                    )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted


    def summarize(self, ax=None, quant: str="time", treat_nan=None,
                  ax_props=None, contour_props=None) -> Tuple:
        """Generates a 2D plot of property values ("time", "noise",
        "n_clusters", "largest") against cluster parameters
        radius_cutoff and cnn_cutoff."""

        if len(self.summary) == 0:
            raise LookupError(
"""No cluster result summary present"""
                )

        pivot = self.summary.groupby(
            ["radius_cutoff", "cnn_cutoff"]
            ).mean()[quant].reset_index().pivot(
                "radius_cutoff", "cnn_cutoff"
                )


        X_, Y_ = np.meshgrid(
            pivot.index.values, pivot.columns.levels[1].values
            )

        values_ = pivot.values.T

        if treat_nan is not None:
            values_[np.isnan(values_)] == treat_nan

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

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

        plotted = []

        plotted.append(
            ax.contourf(X_, Y_, values_, **contour_props_defaults)
            )

        ax.set(**ax_props_defaults)

        return fig, ax, plotted

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
                self._children[key]._data, \
                self._children[key]._shape = \
                self._children[key].get_shape(cluster_data)
                self._children[key]._refindex = np.asarray(ref_index)
                self._children[key]._refindex_rel = np.asarray(ref_index_rel)
        return

    def reel(self, deep: int=1):
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

    def pie(self, ax=None, pie_props=None):
        size = 0.2
        radius = 0.22

        if ax is None:
            ax = plt.gca()

        def getpieces(c, pieces=None, level=0, ref="0"):
            if not pieces:
                pieces = {}
            if not level in pieces:
                pieces[level] = {}

            if c._clusterdict:
                ring = {k: len(v) for k, v in c._clusterdict.items()}
                ringsum = np.sum(list(ring.values()))
                ring = {k: v/ringsum for k, v in ring.items()}
                pieces[level][ref] = ring

                if c._children:
                    for number, child in c._children.items():
                        getpieces(
                            child,
                            pieces=pieces,
                            level=level+1,
                            ref=".".join([ref, str(number)])
                        )

            return pieces
        p = getpieces(self)

        ringvalues = []
        for j in range(np.max(list(p[0]['0'].keys())) + 1):
            if j in p[0]['0']:
                ringvalues.append(p[0]['0'][j])


        ax.pie(ringvalues, radius=radius, colors=None,
            wedgeprops=dict(width=size, edgecolor='w'))

        # iterating through child levels
        for i in range(1, np.max(list(p.keys()))+1):
            # what has not been reclustered:
            reclustered = np.asarray(
                [key.rsplit('.', 1)[-1] for key in p[i].keys()]
                ).astype(int)
            # iterate over clusters of parent level
            for ref, values in p[i-1].items():
                # account for not reclustered clusters
                for number in values:
                    if number not in reclustered:
                        p[i][".".join([ref, str(number)])] = {0: 1}

            # iterate over clusters of child level
            for ref in p[i]:
                preref = ref.rsplit('.', 1)[0]
                sufref = int(ref.rsplit('.', 1)[-1])
                p[i][ref] = {
                    k: v*p[i-1][preref][sufref]
                    for k, v in p[i][ref].items()
                }

            ringvalues = []
            for ref in sorted(list(p[i].keys())):
                for j in p[i][ref]:
                    ringvalues.append(p[i][ref][j])

            ax.pie(ringvalues, radius=radius + i*size, colors=None,
            wedgeprops=dict(width=size, edgecolor='w'))

    def clean(self, which='labels'):
        if which == 'labels':
            _labels = self._labels

            # fixing  missing labels
            n_clusters = len(set(_labels))
            for _cluster in range(1, n_clusters):
                if _cluster not in set(_labels):
                    while _cluster not in set(_labels):
                        _labels[_labels > _cluster] -= 1

            # sorting by clustersize
            n_clusters = np.max(_labels)
            frequency_counts = np.asarray([
                len(np.where(_labels == x)[0])
                for x in set(_labels[_labels > 0])
                ])
            old_labels = np.argsort(frequency_counts)[::-1] +1
            proxy_labels = np.copy(_labels)
            for new_label, old_label in enumerate(old_labels, 1):
                proxy_labels[
                    np.where(_labels == old_label)
                    ] = new_label

            self._labels = proxy_labels

        elif which == 'dict':
            raise NotImplementedError()
        else:
            raise ValueError()

    def labels2dict(self):
        """Convert labels to cluster dictionary
        """

        self._clusterdict = defaultdict(SortedList)
        for _cluster in range(np.max(self._labels) +1):
            self._clusterdict[_cluster].update(
                np.where(self._labels == _cluster)[0]
                )

    def dict2labels(self):
        """Convert cluster dictionary to labels
        """

        self._labels = np.zeros(
            np.sum(len(x) for x in self._clusterdict.values())
            )

        for key, value in self._clusterdict.items():
                self._labels[value] = key


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

        _shape = self.__shape
        _labels = self._labels

        # TODO: Better use numpy split()?
        part_startpoint = 0
        for part in range(0, _shape['parts']):
            part_endpoint = part_startpoint \
                + _shape['points'][part]

            _dtrajs.append(_labels[part_startpoint : part_endpoint])

            part_startpoint = np.copy(part_endpoint)

        return _dtrajs


class CNNChild(CNN):
    """CNN cluster object subclass. Increments the hierarchy level of
    the parent object when instanciated."""

    def __init__(self, parent, alias='child'):
        super().__init__()
        self.hierarchy_level = parent.hierarchy_level +1
        self.alias = alias


def get_histogram(
        x: Sequence[float], y: Sequence[float],
        mids: bool = True, mass: bool = True,
        avoid_zero_count: bool = True,
        hist_props: Optional[Dict['str', Any]] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a two-dimensional histogram.  Taken and modified from
    pyemma.plots.

    Args:
        x: Sample x-coordinates.
        y: Sample y-coordinates.
        hist_props: Kwargs passed to numpy.histogram2d
        avoid_zero_count: Avoid zero counts by lifting all histogram
            elements to the minimum value before computing the free
            energy.  If False, zero histogram counts yield infinity in
            the free energy.
        mass: Norm the histogram by the total number of counts, so that
            each bin holds the probability mass values where all
            probabilities sum to 1
        mids: Return the mids of the bin edges instead of the actual
            edges

    Returns:
        The x- and y-edges and the data of the computed histogram

    """

    hist_props_defaults = {
        'bins': 100,
    }

    if hist_props is not None:
        hist_props_defaults.update(hist_props)

    z, x_, y_ = np.histogram2d(
        x, y, **hist_props_defaults
        )

    if mids:
        x_ = 0.5 * (x_[:-1] + x_[1:])
        y_ = 0.5 * (y_[:-1] + y_[1:])

    if avoid_zero_count:
        z = np.maximum(z, np.min(z[z.nonzero()]))

    if mass:
        z /= float(z.sum())

    return x_, y_, z.T  # transpose to match x/y-directions


def TypedDataFrame(columns, dtypes, content=None, index=None):
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
    # Namemangling allows different defaults in subclasses
    __defaults = {
        'record_points': "points",
        'record_radius_cutoff': "radius_cutoff",
        'record_cnn_cutoff': "cnn_cutoff",
        'record_member_cutoff': "member_cutoff",
        'record_max_cluster': "max_cluster",
        'record_n_cluster': "n_cluster",
        'record_largest': "largest",
        'record_noise': "noise",
        'record_time': "time",
        'default_cnn_cutoff': "1",
        'default_cnn_offset': "0",
        'default_radius_cutoff': "1",
        'default_member_cutoff': "1",
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

    def configure(self, path: Optional[Union[Path, str]] = None,
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
