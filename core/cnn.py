#!/usr/bin/env python

"""This is the core.cnn module - A Python module for the common-nearest-
neighbour (CNN) cluster algorithm.

The functionality provided in this module is based on code implemented
by Oliver Lemke in the script collection CNNClustering available on
git-hub (https://github.com/BDGSoftware/CNNClustering.git). Please cite:

    B. Keller, X. Daura, W. F. van Gunsteren J. Chem. Phys., 2010, 132, 074110.
    O. Lemke, B.G. Keller, J. Chem. Phys., 2016, 145, 164104.
    O. Lemke, B.G. Keller, Algorithms, 2018, 11, 19.
"""

from collections import defaultdict, namedtuple
import warnings
import random
# import json  # unused
from functools import wraps
import time
import pickle
import tempfile
from pathlib import Path
from configparser import ConfigParser
from typing import List, Dict, Tuple, Sequence
from typing import Union, Optional, Type

# from cycler import cycler
# from itertools import cycle, islice

import numpy as np
import pandas as pd  # TODO make this dependency optional?
import matplotlib as mpl
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from scipy.spatial.distance import cdist
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import colorama
import tqdm

# import pyximport
# pyximport.install()
# import .c.cfit


def configure():
    """Read options from configuration file
    """

    CWD = Path.cwd()
    CWD_CONFIG = Path(f"{CWD}/.cnnrc")
    HOME = Path.home()
    HOME_CONFIG = Path(f"{HOME}/.cnnrc")

    config_ = ConfigParser(default_section="settings")
    config_template = ConfigParser(
        default_section="settings",
        defaults={
            'record_points': "points",
            'record_radius_cutoff': "radius_cutoff",
            'record_cnn_cutoff': "cnn_cutoff",
            'record_member_cutoff': "member_cutoff",
            'record_max_cluster': "max_cluster",
            'record_n_cluster': "n_cluster",
            'record_largest': "largest",
            'record_noise': "noise",
            'record_time': "time",
            'color': """
                #000000 #396ab1 #da7c30 #3e9651 #cc2529 #535154
                #6b4c9a #922428 #948b3d #7293cb #e1974c #84ba5b
                #d35e60 #9067a7 #ab6857 #ccc210 #808585
                """,
            'default_cnn_cutoff': "1",
            'default_cnn_offset': "0",
            'default_radius_cutoff': "1",
            'default_member_cutoff': "1",
            'float_precision': 'sp',
            'int_precision': 'sp',
            }
        )

    if CWD_CONFIG.is_file():
        print(f"Configuration file found in {CWD}")
        config_.read(CWD_CONFIG)
    elif HOME_CONFIG.is_file():
        print(f"Configuration file found in {HOME}")
        config_.read(HOME_CONFIG)
    else:
        print("No user configuration file found. Using default setup")
        config_ = config_template
        try:
            with open(HOME_CONFIG, 'w') as configfile:
                config_.write(configfile)
            print(f"Writing configuration file to {HOME_CONFIG}")
        except PermissionError:
            print(
    f"Attempt to write configuration file to {HOME_CONFIG} failed: \
      Permission denied!"
            )
        except FileNotFoundError:
            print(
    f"Attempt to write configuration file to {HOME_CONFIG} failed: \
      No such file or directory!"
            )

    global settings
    global defaults

    settings = config_['settings']
    defaults = config_template['settings']

    # TODO Reconsider use of precision
    global float_precision
    global int_precision

    float_precision = settings.get(
        'float_precision', defaults.get('float_precision')
        )

    int_precision = settings.get(
        'int_precision', defaults.get('int_precision')
        )


# TODO Make this optional
# not really usable since numpy/scipy calculations are done in dp anyways
float_precision_map = {
    'hp': np.float16,
    'sp': np.float32,
    'dp': np.float64,
}

int_precision_map = {
    'qp': np.int8,
    'hp': np.int16,
    'sp': np.int32,
    'dp': np.int64,
}


def timed(function_):
    """Decorator to measure execution time.  Forwards the output of the
       wrapped function and measured excecution time."""
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
    f"Execution time for call of {function_.__name__}(): \
    {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds")
            return wrapped, stopped
    return wrapper


def recorded(function_):
    """Decorator to format function feedback.  Feedback needs to be
       pandas series in record format.  If execution time was measured,
       this will be included in the summary."""

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


class CNN():
    """CNN cluster object class"""

    @staticmethod
    def get_shape(data):
        """Analyses the format of given data and fits it into standard
        format (parts, points, dimensions)."""
        if data is None:
            return None, {
                'parts': None,
                'points': None,
                'dimensions': None,
                }
        else:
            data_shape = np.shape(data[0])
            if np.shape(data_shape)[0] == 0:
                data = np.array([[data]])
                return data, {
                    'parts': 1,
                    'points': [np.shape(data)[0]],
                    'dimensions': 1,
                    }

            elif np.shape(data_shape)[0] == 1:
                data = np.array([data])
                return data, {
                    'parts': 1,
                    'points': [np.shape(data)[1]],
                    'dimensions': np.shape(data)[2],
                    }

            elif np.shape(data_shape)[0] == 2:
                # List of parts passed
                data = np.array([np.asarray(x) for x in data])
                return data, {
                    'parts': np.shape(data)[0],
                    'points': [np.shape(x)[0] for x in data],
                    'dimensions': data_shape[1],
                    }
            else:
                raise ValueError(
f"Data shape {data_shape} not allowed"
                    )

    # TODO Add precision argument on initilisation?
    def __init__(self, alias='root', train=None, test=None,
                 train_dist_matrix=None, test_dist_matrix=None,
                 map_matrix=None):

        self.__alias = alias

        # TODO rather a class attribute?
        self.__hierarchy_level = 0

        if self.__hierarchy_level == 0:
            configure()

        # generic function feedback data container for CCN.cluster(); used only
        # to provide column identifiers.  maybe not too useful ...
        # TODO maybe put this module wide not as instance attribute?
        self.record = namedtuple(
            'ClusterRecord',
            [settings.get('record_points',
             defaults.get('record_points', 'points')),
             settings.get('record_radius_cutoff',
             defaults.get('record_radius_cutoff', 'radius_cutoff')),
             settings.get('record_cnn_cutoff',
             defaults.get('record_cnn_cutoff', 'cnn_cutoff')),
             settings.get('record_member_cutoff',
             defaults.get('record_member_cutoff', 'member_cutoff')),
             settings.get('record_max_clusters',
             defaults.get('record_max_clusters', 'max_clusters')),
             settings.get('record_n_clusters',
             defaults.get('record_n_clusters', 'n_clusters')),
             settings.get('record_largest',
             defaults.get('record_largest', 'largest')),
             settings.get('record_noise',
             defaults.get('record_noise', 'noise')),
             settings.get('record_time',
             defaults.get('record_time', 'time')), ]
            )

        self.__record_dtypes = [
            pd.Int64Dtype(), np.float64, pd.Int64Dtype(), pd.Int64Dtype(),
            pd.Int64Dtype(), pd.Int64Dtype(), np.float64, np.float64,
            np.float64
            ]

        self.test = test
        self.train = train
        # self.__train_stacked
        # self.__test_stacked
        self.__train_dist_matrix = train_dist_matrix
        self.__test_dist_matrix =  test_dist_matrix
        self.__map_matrix = map_matrix
        self.__test_clusterdict = None
        self.__test_labels = None
        self.__train_clusterdict = None
        self.__train_labels = None
        self.summary = TypedDataFrame(
            columns=self.record._fields,
            dtypes=self.__record_dtypes
            )
        self.__train_children = None
        self.__train_refindex = None
        self.__train_refindex_rel = None
        self.__train_tree = None
        self.__test_tree = None
        self.__memory_assigned = None
        self.__cache = None
        # No children for test date. (Hierarchical) clustering should be
        # done on train data

    @property
    def alias(self):
        return self.__alias

    @alias.setter
    def alias(self, a):
        self.__alias = f"{a}"

    @property
    def hierarchy_level(self):
        return self.__hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, level):
        self.__hierarchy_level = int(level)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, data):
        # TODO control string, array, hdf5 file object handling
        self.__test, self.__test_shape = self.get_shape(data)

    @property
    def train(self):
        return self.__train

    @train.setter
    def train(self, data):
        # TODO control string, array, hdf5 file object handling
        self.__train, self.__train_shape = self.get_shape(data)

    @property
    def test_dist_matrix(self):
        return self.__test_dist_matrix

    @test_dist_matrix.setter
    def test_dist_matrix(self, data):
        # TODO control string, array, hdf5 file object handling
        self.__test_dist_matrix = data

    @property
    def train_dist_matrix(self):
        return self.__train_dist_matrix

    @train_dist_matrix.setter
    def train_dist_matrix(self, data):
        # TODO control string, array, hdf5 file object handling
        self.__train_dist_matrix = data

    @property
    def test_shape(self):
        return self.__test_shape

    @test_shape.setter
    def test_shape(self, shape):
        self.__test_shape = shape

    @property
    def train_shape(self):
        return self.__train_shape

    @train_shape.setter
    def train_shape(self, shape):
        self.__train_shape = shape

    @property
    def test_clusterdict(self):
        return self.__test_clusterdict

    @test_clusterdict.setter
    def test_clusterdict(self, d):
        self.__test_clusterdict = d

    @property
    def train_clusterdict(self):
        return self.__train_clusterdict

    @train_clusterdict.setter
    def train_clusterdict(self, d):
        self.__train_clusterdict = d

    @property
    def test_labels(self):
        return self.__test_labels

    @test_labels.setter
    def test_labels(self, d):
        self.__test_labels = d

    @property
    def map_matrix(self):
        return self.__map_matrix

    @property
    def train_labels(self):
        return self.__train_labels

    @train_labels.setter
    def train_labels(self, d):
        self.__train_labels = d

    @property
    def train_tree(self):
        return self.__train_tree

    @property
    def test_tree(self):
        return self.__test_tree

    @property
    def memory_assigned(self):
        return self.__memory_assigned

    @memory_assigned.setter
    def memory_assigned(self, mem):
        self.__memory_assigned = mem

    # @property
    # def summary(self):
    #     return self.__summary

    # @sumary.setter
    # def summary(self, sum):
    #     self.__train_labels = d

    # No setter for summary. This should not be modified by the user.
    # TODO Maybe remove the setter for other critical attributes as well.
    # (shape, clusterdict, labels, children ...) -> may not work

    @property
    def train_children(self):
        return self.__train_children

    @property
    def train_refindex(self):
        return self.__train_refindex

    @property
    def train_refindex_rel(self):
        return self.__train_refindex_rel

    def check(self):
        if self.__test is not None:
            self.test_present = True
            self.test_shape_str = {**self.test_shape}
            self.test_shape_str['points'] = self.test_shape_str['points'][:5]
            if len(self.test_shape['points']) > 5:
                self.test_shape_str['points'] += ["..."]
        else:
            self.test_present = False
            self.test_shape_str = {"parts": None, "points": None, "dimensions": None}

        if self.train is not None:
            self.train_present = True
            self.train_shape_str = {**self.train_shape}
            self.train_shape_str['points'] = self.train_shape_str['points'][:5]
            if len(self.train_shape['points']) > 5:
                self.train_shape_str['points'] += ["..."]
        else:
            self.train_present = False
            self.train_shape_str = {"parts": None, "points": None, "dimensions": None}

        if self.train_dist_matrix is not None:
            self.train_dist_matrix_present = True
        else:
            self.train_dist_matrix_present = False

        if self.test_dist_matrix is not None:
            self.test_dist_matrix_present = True
        else:
            self.test_dist_matrix_present = False

        if self.train_clusterdict is not None:
            self.clusters_present = True
        else:
            self.clusters_present = False

        if self.train_children is not None:
            self.children_present = True
        else:
            self.children_present = False

    def __str__(self):
        self.check()

        return f"""{colorama.Fore.BLUE}cnn.CNN cluster object{colorama.Fore.RESET}
--------------------------------------------------------------------------------
alias :                                  {self.alias}
hierachy level :                         {self.hierarchy_level}

test data shape :                        Parts      - {self.test_shape_str["parts"]}
                                         Points     - {self.test_shape_str["points"]}
                                         Dimensions - {self.test_shape_str["dimensions"]}

train data shape :                       Parts      - {self.train_shape_str["parts"]}
                                         Points     - {self.train_shape_str["points"]}
                                         Dimensions - {self.train_shape_str["dimensions"]}

distance matrix calculated (train) :     {self.train_dist_matrix_present}
distance matrix calculated (test) :      {self.test_dist_matrix_present}
clustered :                              {self.clusters_present}
children :                               {self.children_present}
--------------------------------------------------------------------------------
"""

    def load(self, file_, mode='train', **kwargs):
        """Loads file content and returns data and shape
        """
        # add load option for dist_matrix, map_matrix

        extension = Path(file_).suffix

        case_ = {
            '.p' : lambda: pickle.load(
                open(file_, 'rb'),
                **kwargs
                ),
            '.npy': lambda: np.load(
                file_,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             '': lambda: np.loadtxt(
                 file_,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             '.xvg': lambda: np.loadtxt(
                 file_,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
            '.dat': lambda: np.loadtxt(
                file_,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             }
        data = case_.get(
            extension,
            lambda: print(f"Unknown filename extension {extension}")
            )()

        if mode == 'train':
            self.__train, self.__train_shape = self.get_shape(data)
        elif mode == 'test':
            self.__test, self.__test_shape = self.get_shape(data)
        else:
            raise ValueError(
                "Mode not understood. Only 'train' or 'test' allowed"
                            )

    def delete(self, mode='train'):
        if mode == 'train':
            self.__train = None
        elif mode == 'test':
            self.__test = None
        else:
            raise ValueError(
                "Mode not understood. Only 'train' or 'test' allowed"
                            )

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

    def switch_data(self):
        self.__train, self.__test = self.__test, self.__train
        self.__train_shape, self.__test_shape = self.__test_shape, self.__train_shape
        self.__train_dist_matrix, self.__test_dist_matrix = \
            self.__test_dist_matrix, self.__train_dist_matrix

    def cut(self, parts=(None, None, None), points=(None, None, None),
            dimensions=(None, None, None)):
        """Alows data set reduction.  For each data set level (parts,
        points, dimensions) a tuple (start:stop:step) can be
        specified."""

        if (self.__test is None) and (self.__train is not None):
            print(
                "No test data present, but train data found. Switching data."
                 )
            self.switch_data()
        elif self.__train is None and self.__test is None:
            raise LookupError(
                "Neither test nor train data present."
                )

        self.__train = [x[slice(*points), slice(*dimensions)]
                        for x in self.__test[slice(*parts)]]

        self.__train, self.__train_shape = self.get_shape(self.__train)

    @timed
    def dist(self, mode='train',  v=True, method='cdist', mmap=False,
             mmap_file=None, chunksize=10000, progress=True,):
        """Computes a distance matrix (points x points) for points in given data
        of standard shape (parts, points, dimensions)"""

        if (self.__train is None) and (self.__test is not None):
            print(
                "No train data present, but test data found. Switching data."
                 )
            self.switch_data()
        elif (self.__test is None) and (self.__train is None):
            raise LookupError(
                "Neither test nor train data present."
                )

        if mode == 'train':
            data = self.__train
        elif mode == 'test':
            data = self.__test
        else:
            raise ValueError(
f"Mode {mode} not understood. Must be one of 'train' or 'test'."
                )

        progress = not progress

        if method == 'cdist':
            points = np.vstack(data) # Data can not be streamed right now
            if mmap:
                if mmap_file is None:
                    mmap_file = tempfile.TemporaryFile()

                len_ = len(points)
                _distance_matrix = np.memmap(
                            mmap_file,
                            dtype=float_precision_map[float_precision],
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
                        )
                    ):
                    _distance_matrix[chunk*chunksize: (chunk+1)*chunksize] = cdist(
                        points[chunk*chunksize: (chunk+1)*chunksize], points
                        )
            else:
                _distance_matrix = cdist(points, points)

        else:
            raise ValueError(
f"Method {method} not understood. Must be one of 'cdist' or ... ."
        )

        if mode == 'train':
            self.__train_dist_matrix = _distance_matrix
        elif mode == 'test':
            self.__test_dist_matrix = _distance_matrix

    @timed
    def map(self, method='cdist', mmap=False,
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
                            dtype=float_precision_map[float_precision],
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
            raise ValueError(
f"Method {method} not understood. Must be one of 'cdist' or ... ."
        )

    def dist_hist(self, ax=None, mode="train", maxima=False, maxima_props=None, hist_props=None,
                  ax_props=None, inter_props=None, **kwargs):
        """Shows/saves a histogram plot for distances in a given distance
        matrix"""

    # TODO Add option for kernel density estimation
    # (scipy.stats.gaussian_kde, statsmodels.nonparametric.kde)

        if mode == 'train':
            if self.train_dist_matrix is None:
                print(
"Train distance matrix not calculated. Calculating distance matrix."
                )
                self.dist(mode=mode, **kwargs)
            _dist_matrix = self.train_dist_matrix

        elif mode == 'test':
            if self.test_dist_matrix is None:
                print(
"Test distance matrix not calculated. Calculating distance matrix."
                )
                self.dist(mode=mode, **kwargs)
            _dist_matrix = self.test_dist_matrix
        else:
            raise ValueError(
                "Mode not understood. Must be either 'train' or 'test'."
            )

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
    def fit(self, radius_cutoff: Optional[float]=None, cnn_cutoff: Optional[int]=None,
                member_cutoff: int=None, max_clusters: Optional[int]=None,
                cnn_offset: int=None,
                rec: bool=True, v=True) -> Optional[pd.DataFrame]:
        """Performs a CNN clustering of points in a given train
        distance matrix"""
        # go = time.time()
        # print("Function called")

        if radius_cutoff is None:
            radius_cutoff = float(settings.get(
                                'default_radius_cutoff',
                                defaults.get('default_radius_cutoff', 1)
                                ))
        if cnn_cutoff is None:
            cnn_cutoff = int(settings.get(
                                'default_cnn_cutoff',
                                defaults.get('default_cnn_cutoff', 1)
                                ))
        if member_cutoff is None:
            member_cutoff = int(settings.get(
                                'default_member_cutoff',
                                defaults.get('default_member_cutoff', 1)
                                ))
        if cnn_offset is None:
            cnn_offset = int(settings.get(
                                'default_cnn_offset',
                                defaults.get('default_cnn_offset', 0)
                                ))

        cnn_cutoff -= cnn_offset
        assert cnn_cutoff >= 0

        if (self.__train is None) and (self.__test is not None):
            print(
                "No train data present, but test data found. Switching data."
                 )
            self.switch_data()
        elif (self.__test is None) and (self.__train is None):
            raise LookupError(
                "Neither test nor train data present."
                )

        if self.__train_dist_matrix is None:
            self.dist()

        # print(f"Data checked: {time.time() - go}")

        n_points = len(self.__train_dist_matrix)
        # calculate neighbour list
        neighbours = np.asarray([
            np.where((x > 0) & (x < radius_cutoff))[0]
            for x in self.__train_dist_matrix
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

        self.train_clusterdict = _clusterdict
        self.train_labels = _labels
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
                [len(self.__train_clusterdict) -1],
                [largest],
                [len(self.__train_clusterdict[0]) / n_points],
                [None],
                ],
            )

        if v:
            print(
"\n--------------------------------------------------------------------------------"
            )
            print(cresult[list(self.record._fields)[:-1]].to_string(
                na_rep="None", index=False, line_width=80,
                header=["  #points  ", "  R  ", "  N  ", "  M  ",
                        "  max  ", "  #clusters  ", "  %largest  ", "  %noise  "],
                justify="center"
                ))
            print(
"--------------------------------------------------------------------------------"
            )

        if rec:
            return(cresult)


    def merge(self, clusters, mode='train', which='labels'):
        """Merge a list of clusters into one"""
        if len(clusters) < 2:
            raise ValueError("List of cluster needs to habe at least 2 elements")

        if not isinstance(clusters, list):
            clusters = list(clusters)
        clusters.sort()

        base = clusters[0]

        if which == "labels":
            if mode == 'train':
                _labels = self.__train_labels
            elif mode == 'test':
                _labels = self.__test_labels
            else:
                raise ValueError(f'Mode "{mode}" not understood')

            for add in clusters[1:]:
                _labels[_labels == add] = base

            if mode == 'train':
                self.__train_labels = _labels
            elif mode == 'test':
                self.__test_labels = _labels

            self.clean(mode=mode)
            self.labels2dict(mode=mode)

        elif which == "dict":
            raise NotImplementedError()

            if mode == 'train':
                dict_ = self.__train_clusterdict
            elif mode == 'test':
                dict_ = self.__test_clusterdict
                warnings.warn('Mode "test" not fully functional now' , UserWarning)
            else:
                raise ValueError(f'Mode "{mode}" not understood')

            for add in clusters[1:]:
                dict_[base].update(dict_[add])
                del dict_[add]

            self.clean(mode=mode)
            self.dict2labels(mode=mode)

        else:
            raise ValueError()


        return

    def trash(self, clusters, mode='train', which='labels'):
        """Merge a list of clusters into noise"""

        if which == "labels":
            if mode == 'train':
                _labels = self.__train_labels
            elif mode == 'test':
                _labels = self.__test_labels
            else:
                raise ValueError(f'Mode "{mode}" not understood')

            for add in clusters:
                _labels[_labels == add] = 0

            if mode == 'train':
                self.__train_labels = _labels
            elif mode == 'test':
                self.__test_labels = _labels

            self.clean(mode=mode)
            self.labels2dict(mode=mode)

        elif which == "dict":
            raise NotImplementedError()

            if mode == 'train':
                dict_ = self.__train_clusterdict
            elif mode == 'test':
                dict_ = self.__train_clusterdict
                warnings.warn('Mode "test" not fully functional now' , UserWarning)
            else:
                raise ValueError(f'Mode "{mode}" not understood')


            for cluster in clusters:
                dict_[0].update(dict_[cluster])
                del dict_[cluster]

            self.clean(mode=mode)
            self.dict2labels(mode=mode)

        else:
            raise ValueError()


    def kdtree(self, mode="train", **kwargs):
        """CNN.method wrapper for scipy.spatial.cKDTree
        """
        if mode == "train":
            self.__train_tree = cKDTree(np.vstack(self.__train), **kwargs)
        elif mode == "test":
            self.__test_tree = cKDTree(np.vstack(self.__test), **kwargs)

    @staticmethod
    def get_neighbours(a: Type[np.ndarray], B: Type[np.ndarray],
                       r: float) -> List[int]:
        """Returns a list of indeces of points in B that are neighbours
        of a within radius r."""
        # r = r**2
        return np.where(np.sum((B - a)**2, axis=1) < r)[0]

    @timed
    def predict(self, radius_cutoff: Optional[float]=None,
        cnn_cutoff: Optional[int]=None, member_cutoff: Optional[int]=None,
        include_all: bool=True, same_tol=1e-8, memorize: bool=True,
        clusters: Optional[List[int]]=None, purge: bool=False,
        cnn_offset: Optional[int]=None, behaviour="lookup",
        method='plain', progress=True, **kwargs) -> None:
        """
        Predict labels for points in a test set on the basis of assigned
        labels to a train set by :method:`CNN.fit`

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
            that have a maximum distance of :param:`same_tol` to a point
            in the train set, i.e. they are (essentially the same point)
            (currently not implemented)

        same_tol : float, default=1e-8
            Distance cutoff to treat points as the same, if
            :param:`include_all` is False

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
                throw an error.  Consider memory mapping :param:`mmap`
                when computing the distances with :method:`CNN.dist` and
                :method:`CNN.map` for large data sets.

            * "on-the-fly", Compute distances during the prediction
                using the specified :param:`method`.

            * "tree", Get the neighbour lists during the prediction from
                a tree query

        method : str, default="plain"
            Controlls which method is used to get the neighbour lists
            within a given :param:`behaviour`:

            * "lookup", parameter not used

            * "on-the-fly",
                * "plain", uses :method:`CNN.get_neighbours`

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
            radius_cutoff = float(settings.get(
                'default_radius_cutoff',
                defaults.get('default_radius_cutoff', 1.)
                ))

        if cnn_cutoff is None:
            cnn_cutoff = int(settings.get(
                'default_cnn_cutoff',
                defaults.get('default_cnn_cutoff', 1)
                ))

        if member_cutoff is None:
            member_cutoff = int(settings.get(
                'default_member_cutoff',
                defaults.get('default_member_cutoff', 1)
                ))

        if cnn_offset is None:
            cnn_offset = int(settings.get(
                'default_cnn_offset',
                defaults.get('default_cnn_offset', 0)
                ))

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
    def check_similarity(a: List[int], b: List[int], c: int) -> bool:
        """Returns True if list a and list b have at least c common elements
        """
        # Overlapping train and test sets
        #  if len(set(a).intersection(b)) - 1 >= c:
        if len(set(a).intersection(b)) >= c:
           return True

    @staticmethod
    def check_similarity_array(a: Type[np.ndarray], b: Type[np.ndarray], c: int) -> bool:
        """Returns True if list a and list b have at least c common elements
        """
        # Overlapping train and test sets
        #  if len(np.intersect1d(a, b, assume_unique=True)) - 1 >= c:
        if len(np.intersect1d(a, b, assume_unique=True)) >= c:
           return True

    def query_data(self, mode='train'):
        """Helper function to evaluate user input. If data is required as
        keyword argument and data=None is passed, the default data used is
        either self.rdata or self.data."""

        if mode == 'train':
            if self.train is not None:
                _data = self.train
                _shape = self.train_shape
            elif self.test is None:
                raise LookupError(
                    "No data available"
                    )
            else:
                _data = self.test
                _shape = self.test_shape
        elif mode == 'test':
            if self.test is not None:
                _data = self.test
                _shape = self.test_shape
            elif self.train is None:
                raise LookupError(
                    "No data available"
                    )
            else:
                _data = self.train
                _shape = self.train_shape

        return _data, _shape

    def evaluate(
        self,
        ax: Optional[Type[mpl.axes._subplots.AxesSubplot]] = None,
        mode: str='train', clusters: Optional[List[int]]=None,
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
        ):

        """Returns a 2D plot of an original data set or a cluster result

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, default=None
            The axes to which to add the plot.  If None, a new figure
            with axes will be created.

        mode : str, default="train"
            Which data ("train" or "test") to use for this plot

        clusters : List[int], default=None
            Cluster numbers to include in the plot.  If None, consider
            all.

        original : bool, default=False
            Allows to plot the original data instead of a cluster
            result.  Overrides :param:`clusters`.  Will be considered
            True, if no cluster result is present.

        plot : str, default="dots"
            The kind of plotting method to use.

            * "dots", Use :function:`ax.plot()`

            * "",

        parts : Tuple[int, int, int] (length 3), default=(None, None, None)
            Use a slice (start, stop, stride) on the data parts before
            plotting.

        points : Tuple[int, int, int], default=(None, None, None)
            Use a slice (start, stop, stride) on the data points before
            plotting.

        dim : Tuple[int, int], default=None
            Use these two dimensions for plotting.  If None, uses
            (0, 1).

        annotate : bool, default=True
            If there is a cluster result, plot the cluster numbers.  Uses
            :param:`annotate_pos` to determinte the position of the
            annotations.

        annotate_pos : str or List[Tuple[int, int]], default="mean"
            Where to put the cluster number annotation.  Can be one of:

            * "mean", Use the cluster mean

            * "random", Use a random point of the cluster

            Alternatively a list of x, y positions can be passed to set
            a specific point for each cluster (Not yet implemented)

        annotate_props : Dict, default=None
            Dictionary of keyword arguments passed to
            :function:`ax.annotate(**kwargs)`.

        ax_props : Dict, default=None
            Dictionary of :param:`ax` properties to apply after
            plotting via :function:`ax.set(**ax_props)`.  If None, uses
            defaults that can be also defined in the configuration file.

        (hist, contour, dot, scatter, dot_noise, scatter_noise)_props : Dict, default=None
            Dictionaries of keyword arguments passed to various
            functions.  If None, uses
            defaults that can be also defined in the configuration file.

        mask : Sequence[bool]
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

        if mode == 'train':
            try:
                items = self.__train_clusterdict.items()
                if clusters is None:
                    clusters = list(range(len( items )))
            except AttributeError:
                original = True

        elif mode == 'test':
            try:
                items = self.__test_clusterdict.items()
                if clusters is None:
                    clusters = list(range(len( items )))
            except AttributeError:
                original = True
        else:
            raise ValueError()

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

    def isolate(self, mode='train', purge=True):
        """Isolates points per clusters based on a cluster result"""

        if mode == 'train':
            if purge or self.__train_children is None:
                self.__train_children = defaultdict(lambda: CNNChild(self))

            for key, _cluster in  self.__train_clusterdict.items():
                # TODO: What if no noise?
                if len(_cluster) > 0:
                    _cluster = np.asarray(_cluster)
                    ref_index = []
                    ref_index_rel = []
                    cluster_data = []
                    part_startpoint = 0

                    if self.__train_refindex is None:
                        ref_index.extend(_cluster)
                        ref_index_rel = ref_index
                    else:
                        ref_index.extend(self.__train_refindex[_cluster])
                        ref_index_rel.extend(_cluster)

                    for part in range(self.__train_shape['parts']):
                        part_endpoint = part_startpoint \
                            + self.__train_shape['points'][part] -1

                        cluster_data.append(
                            self.__train[part][_cluster[
                                np.where(
                                    (_cluster
                                    >= part_startpoint)
                                    &
                                    (_cluster
                                    <= part_endpoint))[0]] - part_startpoint]
                                )
                        part_startpoint = np.copy(part_endpoint)
                        part_startpoint += 1

                    self.__train_children[key].__alias = f'child No. {key}'
                    self.__train_children[key].__train, \
                    self.__train_children[key].__train_shape = \
                    self.__train_children[key].get_shape(cluster_data)
                    self.__train_children[key].__train_refindex = np.asarray(
                                                                       ref_index
                                                                       )
                    self.__train_children[key].__train_refindex_rel = np.asarray(
                                                                       ref_index_rel
                                                                       )
        else:
            raise NotImplementedError()


    def reel(self, deep=1):
        if self.__train_children is None:
            raise LookupError(
                "No child clusters isolated"
                             )
        # TODO: Implement "deep" for degree of decent into hierarchy structure

        for _cluster in self.__train_children.values():
            n_clusters = max(self.__train_clusterdict)
            if _cluster.__train_labels is not None:
                if self.hierarchy_level == 0:
                    self.__train_labels[
                    _cluster.__train_refindex[
                        np.where(_cluster.__train_labels == 0)[0]
                        ]
                    ] = 0
                else:
                    self.__train_labels[
                    _cluster.__train_refindex_rel[
                        np.where(_cluster.__train_labels == 0)[0]
                        ]
                    ] = 0

                for _label in _cluster.__train_labels[_cluster.__train_labels > 1]:
                    if self.hierarchy_level == 0:
                        self.__train_labels[
                        _cluster.__train_refindex[
                            np.where(_cluster.__train_labels == _label)[0]
                            ]
                        ] = _label + n_clusters
                    else:
                        self.__train_labels[
                        _cluster.__train_refindex_rel[
                            np.where(_cluster.__train_labels == _label)[0]
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

            if c.train_clusterdict:
                ring = {k: len(v) for k, v in c.train_clusterdict.items()}
                ringsum = np.sum(list(ring.values()))
                ring = {k: v/ringsum for k, v in ring.items()}
                pieces[level][ref] = ring

                if c.train_children:
                    for number, child in c.train_children.items():
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

    def clean(self, which='labels', mode='train'):
        if which == 'labels':
            if mode == 'train':
                _labels = self.__train_labels
            elif mode == 'test':
                _labels = self.__test_labels
            else:
                raise ValueError()

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

            if mode == 'train':
                self.__train_labels = proxy_labels
            elif mode == 'test':
                self.__test_labels = proxy_labels

        elif which == 'dict':
            raise NotImplementedError()
        else:
            raise ValueError()

    def labels2dict(self, mode='train'):
        if mode == 'train':
            self.__train_clusterdict = defaultdict(SortedList)
            for _cluster in range(np.max(self.__train_labels) +1):
                self.__train_clusterdict[_cluster].update(
                    np.where(self.__train_labels == _cluster)[0]
                    )

                """
                if self.train_refindex is None:
                    self.train_clusterdict[_cluster].extend(
                        np.where(self.train_labels == _cluster)[0]
                        )
                else:
                    self.train_clusterdict[_cluster].extend(
                        self.train_refindex[
                        np.where(self.train_labels == _cluster)[0]
                        ])
                """
        elif mode == 'test':
            self.__test_clusterdict = defaultdict(SortedList)
            for _cluster in range(np.max(self.__test_labels) +1):
                self.__test_clusterdict[_cluster].update(
                    np.where(self.__test_labels == _cluster)[0]
                    )
        else:
            raise NotImplementedError()

    def dict2labels(self, mode='train'):
        if mode == 'train':
            self.__train_labels = np.zeros(
                np.sum(len(x) for x in self.__train_clusterdict.values())
                )

            for key, value in self.__train_clusterdict.items():
                self.__train_labels[value] = key
                """
                if self.train_refindex is None:
                    self.train_labels[value] = key
                else:
                    self.train_labels[self.train_refindex[value]] = key
                """
        else:
            raise NotImplementedError()


    def get_samples(self, mode='train', kind='mean', clusters=None,
                    n_samples=1, byparts=True, skip=1, stride=1):
        if clusters is None:
            clusters = list(range(1, len()))

        if mode == 'train':
            dict_ = self.__train_clusterdict
            _data = np.vstack(self.__train)
            _shape = self.__train_shape
        elif mode == 'test':
            dict_ = self.__test_clusterdict
            _data = np.vstack(self.__test)
            _shape = self.__test_shape
        else:
            raise ValueError()

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


    def get_dtraj(self, mode='train'):
        _dtrajs = []

        if mode == 'train':
            _shape = self.__train_shape
            _labels = self.__train_labels
        elif mode == 'test':
            _shape = self.__test_shape
            _labels = self.__test_labels
        else:
            raise ValueError()

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
        self.__alias = alias

########################################################################

def get_histogram(x, y, mids=True, mass=True, avoid_zero_count=True, hist_props=None):
    """Compute a two-dimensional histogram.  Taken and modified from
    pyemma.plots.

    Parameters
    ----------
    x : ndarray(T)
        Sample x-coordinates.
    y : ndarray(T)
        Sample y-coordinates.
    hist_props : Dict, optional
        kwargs passed to numpy.histogram2d
    avoid_zero_count : bool, optional, default=True
        Avoid zero counts by lifting all histogram elements to the
        minimum value before computing the free energy. If False,
        zero histogram counts would yield infinity in the free energy.
    mass : bool, default=True
        Norm the histogram by the total number of counts, so that
        each bin holds the probability mass values where all
        probabilities sum to 1
    mids : bool, default=True
       Return the mids of the bin edges instead of the actual edges

    Returns
    -------
    x_ : ndarray(nbins, nbins)
        The bins' x-coordinates in meshgrid format.
    y_ : ndarray(nbins, nbins)
        The bins' y-coordinates in meshgrid format.
    z : ndarray(nbins, nbins)
        Histogram counts in meshgrid format.

    """

    hpt = {
        'bins': 100,
    }

    if hist_props is not None:
        hpt.update(hist_props)

    z, x_, y_ = np.histogram2d(
        x, y, **hpt
        )

    if mids:
        x_ = 0.5 * (x_[:-1] + x_[1:])
        y_ = 0.5 * (y_[:-1] + y_[1:])

        # x_ = x_[:-1] + (x_[-1] - x_[0]) / ((len(x_) - 1)*2)
        # y_ = y_[:-1] + (y_[-1] - y_[0]) / ((len(y_) - 1)*2)

    if avoid_zero_count:
        z = np.maximum(z, np.min(z[z.nonzero()]))

    if mass:
        z /= float(z.sum())

    return x_, y_, z.T # transpose to match x/y-directions

def TypedDataFrame(columns, dtypes, content=None, index=None):
    assert len(columns) == len(dtypes)

    if content is None:
        content = [[] for i in range(len(columns))]

    df = pd.DataFrame({
        k: pd.array(c, dtype=v)
        for k, v, c in zip(columns, dtypes, content)
        })

    return df

def dist(data):
    """High level wrapper function for cnn.CNN().dist(). Takes data and
    returns a distance matrix (points x points).
    """
    cobj = CNN(train=data)
    cobj.dist()
    return cobj.train_dist_matrix

########################################################################

# TODO Alter configuration mechanism to use .yaml?

########################################################################

if __name__ == "__main__":
    configure()
