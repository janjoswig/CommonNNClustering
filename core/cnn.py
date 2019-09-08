"""This is cnn v0.1

The functionality provided in this module is based on code implemented
by Oliver Lemke in the script collection CNNClustering available on
git-hub (https://github.com/BDGSoftware/CNNClustering.git).

Author: Jan-Oliver Joswig, 
first released: 03.12.2018
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict, namedtuple
from sortedcontainers import SortedList
from scipy.spatial.distance import cdist
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

# import pyximport
# pyximport.install()
# import .c.cfit

from functools import wraps
import time
import pandas as pd # TODO get rid of this dependency?
from configparser import ConfigParser
# from cycler import cycler
from itertools import cycle, islice
from pathlib import Path

from typing import List, Dict
from typing import Union, Optional

import warnings

#######################################################################
# TODO Consider two points at the same (within precision) coordinates
# as one or treat seperately? 

########################################################################
# Global functions

def configure():
    """Read from configuration file
    """
    CWD = Path.cwd()
    CWD_CONFIG = Path(f"{CWD}/.cnnrc")
    HOME = Path.home()
    HOME_CONFIG = Path(f"{HOME}/.cnnrc")
    config_ = ConfigParser(default_section="settings")
    config_template = ConfigParser(
            default_section="settings",
            defaults={'record_points': "points",
                    'record_radius_cutoff' : "radius_cutoff",
                    'record_cnn_cutoff' : "cnn_cutoff",
                    'record_member_cutoff' : "member_cutoff",
                    'record_max_cluster' : "max_cluster",
                    'record_n_cluster' : "n_cluster",
                    'record_largest' : "largest",
                    'record_noise' : "noise",
                    'record_time' : "time",
                    'color' : """#000000 #396ab1 #da7c30 #3e9651 #cc2529 #535154
                                 #6b4c9a #922428 #948b3d #7293cb #e1974c #84ba5b
                                 #d35e60 #9067a7 #ab6857 #ccc210 #808585""",
                    'default_cnn_cutoff' : "1",
                    'default_cnn_offset' : "0",
                    'default_radius_cutoff' : "1",
                    'default_member_cutoff' : "1",
                    'float_precision' : 'sp',
                    'int_precision' : 'sp',
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
            if len(wrapped) > 1:
                wrapped[-2]['time'] = wrapped[-1]
                print(f'recording: ... \n{wrapped[-2]}')
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
                    defaults.get('record_time', 'time')),]
                    )
        
        self.__test = test
        self.__train = train
        self.__train_dist_matrix = train_dist_matrix
        self.__test_dist_matrix =  test_dist_matrix
        self.__map_matrix = map_matrix
        self.__test, self.__test_shape = self.get_shape(self.__test)
        self.__train, self.__train_shape = self.get_shape(self.__train)
        self.__test_clusterdict = None
        self.__test_labels = None
        self.__train_clusterdict = None
        self.__train_labels = None
        self.summary = pd.DataFrame(columns=self.record._fields)
        self.__train_children = None
        self.__train_refindex = None
        self.__train_refindex_rel = None
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
        self.__test = data

    @property
    def train(self):
        return self.__train

    @train.setter
    def train(self, data):
        # TODO control string, array, hdf5 file object handling
        self.__train = data

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
    def train_labels(self):
        return self.__train_labels

    @property
    def map_matrix(self):
        return self.__map_matrix    

    @train_labels.setter
    def train_labels(self, d):
        self.__train_labels = d

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
            self.test_shape_str = None

        if self.train is not None:
            self.train_present = True
            self.train_shape_str = {**self.train_shape}
            self.train_shape_str['points'] = self.train_shape_str['points'][:5]
            if len(self.train_shape['points']) > 5:
                self.train_shape_str['points'] += ["..."]    
        else:
            self.train_present = False            
            self.train_shape_str = None

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
        return f"""cnn.CNN() cluster object 
alias :                                 {self.alias}
hierachy level:                         {self.hierarchy_level}
test data loaded :                      {self.test_present}
test data shape :                       {self.test_shape_str}
train data loaded :                     {self.train_present}
train data shape :                      {self.train_shape_str}
distance matrix calculated (train):     {self.train_dist_matrix_present}
distance matrix calculated (test):      {self.test_dist_matrix_present}
clustered :                             {self.clusters_present}
children :                              {self.children_present}
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
            self.train, self.train_shape = self.get_shape(data)
        elif mode == 'test':
            self.test, self.test_shape = self.get_shape(data)
        else:
            raise ValueError(
                "Mode not understood. Only 'train' or 'test' allowed"
                            )

    def delete(self, mode='train'):
        if mode == 'train':
            self.train = None
        elif mode == 'test':
            self.test = None
        else:
            raise ValueError(
                "Mode not understood. Only 'train' or 'test' allowed"
                            )

    def save(self, file_, content):
        """Saves content to file"""

        extension = file_.rsplit('.', 1)[-1]
        if len(extension) == 1:
            extension = ''
        {
        'p' : lambda: pickle.dump(open(file_, 'wb'), content),
        'npy': lambda: np.save(file_, content),
        '': lambda: np.savetxt(file_, content),
        }.get(extension,
            f"Unknown filename extension .{extension}")()

    def switch_data(self):
        self.train, self.test = self.test, self.train
        self.train_shape, self.test_shape = self.test_shape, self.train_shape
        self.train_dist_matrix, self.test_dist_matrix = \
            self.test_dist_matrix, self.train_dist_matrix

    def cut(self, parts=(None, None, None), points=(None, None, None),
            dimensions=(None, None, None)):
        """Alows data set reduction.  For each data set level (parts,
        points, dimensions) a tuple (start:stop:step) can be
        specified."""

        if (self.test is None) and (self.train is not None):
            print(
                "No test data present, but train data found. Switching data."    
                 )
            self.switch_data()
        elif self.train is None and self.test is None:
            raise LookupError(
                "Neither test nor train data present."
                )

        self.train = [x[slice(*points), slice(*dimensions)] 
                        for x in self.test[slice(*parts)]]
    
        self.train, self.train_shape = self.get_shape(self.train)

    @timed
    def dist(self, mode='train',  v=True, low_memory=False):
        """Computes a distance matrix (points x points) for points in given data
        of standard shape (parts, points, dimensions)"""

        if (self.train is None) and (self.test is not None):
            print(
                "No train data present, but test data found. Switching data."    
                 )
            self.switch_data()

        if mode == 'train':
            points = np.vstack(self.train)
            # _dist_matrix = self.train_dist_matrix
        elif mode == 'test':
            points = np.vstack(self.test)
            # _dist_matrix = self.test_dist_matrix
        else:
            raise ValueError(
            "Mode not understood. Must be one of 'train' or 'test'."
            )
        
        if v:
            print(
f"Calculating nxn distance matrix for {len(points)} points"
            )

        if low_memory:
            raise NotImplementedError()
        else:
            #_dist_matrix = cdist(points, points)

            if mode == 'train':
                self.train_dist_matrix = cdist(points, points)
            elif mode == 'test':
                self.test_dist_matrix = cdist(points, points)
                # _dist_matrix = self.test_dist_matrix

    @timed
    def map(self, nearest=None):
        """Computes a map matrix that maps an arbitrary data set to a
        reduced to set"""

        if self.train is None or self.test is None:
            raise LookupError(
                "Mapping requires a train and a test data set"
                )
        elif self.train_shape['dimensions'] != self.test_shape['dimensions']:
            raise ValueError(
                "Mapping requires the same number of dimension in the train \
                 and the test data set"
                )

        if nearest is not None:
            raise NotImplementedError()
        else:
            self.__map_matrix = cdist(np.vstack(self.test), np.vstack(self.train))
            
    def dist_hist(self, ax=None, mode='train', show=True, save=False,
                  output='dist_hist.pdf', maxima=False, hist_props=None,
                  ax_props=None, inter_props=None, save_props=None):
        """Shows/saves a histogram plot for distances in a given distance
        matrix"""
        
    # TODO Add option for kernel density estimation 
    # (scipy.stats.gaussian_kde, statsmodels.nonparametric.kde) 

        if mode == 'train':
            if self.train_dist_matrix is None:
                print(
"Train distance matrix not calculated. Calculating distance matrix."
                )
                self.dist(mode=mode)
            _dist_matrix = self.train_dist_matrix
        
        elif mode == 'test':
            if self.test_dist_matrix is None:
                print(
"Test distance matrix not calculated. Calculating distance matrix."
                )
                self.dist(mode=mode)           
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

        # TODO make this a configuation option
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
            ax = plt.gca()

        ax.plot(binmids, histogram)

        if maxima:
            found = argrelextrema(histogram, np.greater)[0]
            settings['default_radius_cutoff'] = \
                f"{binmids[found[0]]:.2f}"
            for candidate in found:
                ax.annotate(
                    f"{binmids[candidate]:.2f}",
                    xy=(binmids[candidate], histogram[candidate]),
                    xytext=(binmids[candidate], 
                            histogram[candidate]+(ylimit/100))
                    )

        ax.set(**ax_props_defaults)

        # TODO make this a configuation option
        save_props_defaults = {}

        if save_props is not None:
            save_props_defaults.update(save_props)
        
        # make this optional
        plt.tight_layout(pad=0.1)
        if save:
            plt.savefig(output, **save_props_defaults)
        if show:
            plt.show()

    @recorded
    @timed
    def fit(self, radius_cutoff: float=None, cnn_cutoff: int=None,
                member_cutoff: int=None, max_clusters: int=None,
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
            np.where((x > 0) & (x <= radius_cutoff))[0]
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
            # find point with highest neighbour count
            point = np.where(
                (n_neighbours == np.max(n_neighbours[include]))
                & (include == True)
                )[0][0]
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

        if rec:
            # print(f"Returning: {time.time() - go}")
            return pd.Series([
                        n_points,
                        radius_cutoff,
                        cnn_cutoff,
                        member_cutoff,
                        max_clusters,
                        len(self.train_clusterdict) -1,
                        largest,
                        len(self.train_clusterdict[0]) / n_points,
                        None,
                        ],
                        index=self.record._fields,
                        dtype='object',
                        )
    
    def merge(self, clusters, mode='train'):
        """Merge a list of clusters into one"""
        if len(clusters) < 2:
            raise ValueError("List of cluster needs to habe at least 2 elements")
        
        if mode == 'train':
            dict_ = self.__train_clusterdict
        elif mode == 'test':
            dict_ = self.__train_clusterdict
            warnings.warn('Mode "test" not fully functional now' , UserWarning)
        else:
            raise ValueError(f'Mode "{mode}" not understood')

        if not isinstance(clusters, list):
            clusters = list(clusters)
        clusters.sort()
            
        base = clusters[0]
        for add in clusters[1:]:
            dict_[base].update(dict_[add])
            del dict_[add]

        self.clean()
        return


    @timed
    def predict(self, low_memory=False, radius_cutoff=1, cnn_cutoff=1,
        member_cutoff=1, max_clusters=None, include_all=True, cluster=None,
        purge=False):
        
        if low_memory:
            # raise NotImplementedError()
            if cluster is None:
                train_neighbours = np.asarray([                  
                    np.where(x <= radius_cutoff)[0] for x in self.__train_dist_matrix
                    ])
            else:
                selected_members = np.concatenate(
                    [self.__train_clusterdict[x] for x in cluster]
                    )
                all_neighbours = np.asarray([                  
                    np.where(x <= radius_cutoff)[0] for x in self.__train_dist_matrix
                    ])
                train_neighbours = np.asarray([
                    x[np.isin(x, selected_members)] for x in all_neighbours
                    ])
            
            # n_points = np.sum(self.test_shape['points'])
            _cdist = cdist
            test = np.vstack(self.test)
            train = np.vstack(self.train)
            n_points = len(test)

            if (cluster is None) or (purge is True) or (self.__test_labels is None):
                test_labels = np.zeros(n_points).astype(int)
            else:
                test_labels = np.asarray([x if x not in cluster else 0 for x in self.__test_labels])


            if not include_all:
                for point in range(n_points):
                    if point % 5000 == 0: 
                        print(
f'Predicting cluster for point {1+point:>7} of {n_points:<7}',
end='\r'
                            )
                    map_matrix_part = _cdist(np.array([test[point]]),
                                             train)
                     
                    same = np.where(map_matrix_part[0] == 0)[0]
                    if len(same) > 0:
                        test_labels[point] = self.__train_labels[same[0]]
                    else:
                        test_neighbours = np.where(map_matrix_part[0] <= radius_cutoff)[0]
                        if cluster is not None:
                            test_neighbours = test_neighbours[
                                np.isin(test_neighbours, selected_members)]
                       
                        common_neighbours = [
                                    set(test_neighbours)
                                    & set(train_neighbours[x])
                                    for x in test_neighbours
                                    ]

                        cnn_fulfilled = np.where(
                            np.asarray([
                            len(x) for x in common_neighbours
                            ]) >= cnn_cutoff)[0]

                        if len(cnn_fulfilled) > 0:
                            test_labels[point] = self.train_labels[
                            test_neighbours[
                            np.argmax(cnn_fulfilled)]] 
            
            else:             
                for point in range(n_points):
                    if point % 5000 == 0:
                        print(
                        f'Predicting cluster for point {1+point:6} of {n_points}',
                        end='\r'
                        )
                
                    map_matrix_part = _cdist(np.array([test[point]]),
                                             train)
                    test_neighbours = np.where(map_matrix_part[0] <= radius_cutoff)[0]
                    if cluster is not None:
                        test_neighbours = test_neighbours[
                                np.isin(test_neighbours, selected_members)]

                    common_neighbours = [
                                set(test_neighbours)
                                & set(train_neighbours[x])
                                for x in test_neighbours
                                ]

                    cnn_fulfilled = np.where(
                        np.asarray([
                        len(x) for x in common_neighbours
                        ]) >= cnn_cutoff)[0]

                    if len(cnn_fulfilled) > 0:
                        test_labels[point] = self.train_labels[
                        test_neighbours[
                        np.argmax(cnn_fulfilled)]] 


        else:
            if self.__map_matrix is None:
                self.map()
        
            if cluster is None:
                test_neighbours = np.asarray([
                    np.where(x <= radius_cutoff)[0] for x in self.__map_matrix
                    ])
            
                train_neighbours = np.asarray([                  
                    np.where(x <= radius_cutoff)[0] for x in self.train_dist_matrix
                    ])
            else:
                selected_members = np.concatenate(
                    [self.train_clusterdict[x] for x in cluster]
                    )
                all_neighbours = np.asarray([
                    np.where(x <= radius_cutoff)[0] for x in self.__map_matrix
                    ])
                test_neighbours = np.asarray([
                   x[np.isin(x, selected_members)] for x in all_neighbours
                    ])

                all_neighbours = np.asarray([                  
                    np.where(x <= radius_cutoff)[0] for x in self.train_dist_matrix
                    ])
                train_neighbours = np.asarray([
                    x[np.isin(x, selected_members)] for x in all_neighbours
                    ])

            n_points = len(self.__map_matrix)

            if (cluster is None) or (purge is True) or (self.test_labels is None):
                test_labels = np.zeros(n_points).astype(int)
            else:
                test_labels = np.asarray([x if x not in cluster else 0 for x in self.__test_labels])

            if not include_all:
                for point in range(n_points):
                    if point % 5000 == 0:
                        print(f'Predicting cluster for point {1+point:6} of {n_points}',
                            end='\r')
                    same = np.where(self.__map_matrix[point] == 0)[0]
                    if len(same) > 0:
                        test_labels[point] = self.train_labels[same[0]]
                    else:
                        common_neighbours = [
                            set(test_neighbours[point])
                            & set(train_neighbours[x])
                            for x in test_neighbours[point]
                            ]

                        cnn_fulfilled = np.where(
                            np.asarray([
                                len(x) for x in common_neighbours
                                ]) >= cnn_cutoff)[0]
              
                        if len(cnn_fulfilled) > 0:
                            test_labels[point] = self.train_labels[
                                test_neighbours[point][
                                    np.argmax(cnn_fulfilled)]] 
            else:
                for point in range(n_points):
                    if point % 5000 == 0:
                        print(f'Predicting cluster for point {1+point:6} of {n_points}',
                            end='\r')
               
                    common_neighbours = [
                        set(test_neighbours[point])
                        & set(train_neighbours[x])
                        for x in test_neighbours[point]
                        ]

                    cnn_fulfilled = np.where(
                        np.asarray([
                            len(x) for x in common_neighbours
                            ]) >= cnn_cutoff)[0]
              
                    if len(cnn_fulfilled) > 0:
                        test_labels[point] = self.train_labels[
                            test_neighbours[point][
                                np.argmax(cnn_fulfilled)]]

        self.__test_labels = test_labels
        self.labels2dict(mode="test")

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

    def evaluate(self, ax=None, mode='train', max_clusters=None,
                 plot='dots', parts=(None, None, None),
                 points=(None, None, None), dim=None, show=False, save=False,
                 output='evaluation.png', ax_props=None, save_props=None, 
                 scatter_props=None, hist_props=None, dot_props=None, 
                 dot_noise_props=None, scatter_noise_props=None,
                 contour_props=None, original=False, annotate=True):
        """Shows/saves a 2D histogram or scatter plot of a cluster result"""
        
        # BROKEN!!!

        # TODO fix plotting when set is cut
        # TODO make noise optional
        # TODO overlay clusters over noise
        _data, _ = self.query_data(mode=mode)
        if dim is None:
            dim = (0, 1)
        elif dim[1] < dim[0]:
            dim = dim[::-1]

        _data = [
            x[slice(*points), slice(dim[0], dim[1]+1, dim[1]-dim[0])] 
            for x in _data[slice(*parts)]
            ]

        _data = np.vstack(_data)

        # if mode == 'test':
        #     if self.__test_labels is None:
        #         _labels = np.ones(len(_data)).astype(int)
        #     else:
        #         _labels = self.test_labels
        # elif mode == 'train':
        #     if self.__train_labels is None:
        #         _labels = np.ones(len(_data)).astype(int)
        #     else:
        #         _labels = self.train_labels
        if mode == 'train':
            try:
                items = self.__train_clusterdict.items()
                if max_clusters is None:
                    max_clusters = max(self.__train_clusterdict)
            except AttributeError:
                original = True

        elif mode == 'test':
            try:
                items = self.__test_clusterdict.items()
                if max_clusters is None:
                    max_clusters = max(self.__test_clusterdict)
            except AttributeError:
                original = True
        else:
            raise ValueError()


        # if max_clusters is not None:
        #     _labels[_labels > max_clusters] = 0

        # TODO make this a configuation option
        ax_props_defaults = {
            "xlabel": "$x$",
            "ylabel": "$y$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if plot == 'dots':
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
                ax.plot(
                    _data[:, 0],
                    _data[:, 1],
                    **dot_props_defaults
                    )

            else:
                for cluster, points in items:
                    if cluster > max_clusters:
                        break

                    if cluster == 0:
                        ax.plot(
                            _data[points, 0],
                            _data[points, 1],
                            **dot_noise_props_defaults
                        )
                    else:
                        ax.plot(
                            _data[points, 0],
                            _data[points, 1],
                            **dot_props_defaults
                            )
                        if annotate:
                            ax.annotate(
                                f"{cluster}",
                                xy=(np.mean(_data[points, 0]),
                                    np.mean(_data[points, 1])),
                                )

        elif plot == 'scatter':
            # color = settings.get('color', defaults.get('color'))
            # if color is not None:
            #     colors = np.array(
            #         list(islice(cycle(
            #             color.split(' ')
            #                 ),
            #             int(max(_labels) + 1)
            #             ))
            #         ) 

            #    color = colors[_labels]

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
                ax.scatter(
                    _data[:, 0],
                    _data[:, 1],
                    **scatter_props_defaults
                    )

            else:
                for cluster, points in items:
                    if cluster > max_clusters:
                        break

                    if cluster == 0:
                        ax.scatter(
                            _data[points, 0],
                            _data[points, 1],
                            **scatter_noise_props_defaults
                            )
                    else:
                        ax.scatter(
                            _data[points, 0],
                            _data[points, 1],
                            **scatter_props_defaults
                            )
                        if annotate:
                            ax.annotate(
                                f"{cluster}",
                                xy=(np.mean(_data[points, 0]),
                                    np.mean(_data[points, 1])),
                                )

        elif plot in ['contour', 'contourf', 'histogram']:
            # TODO make this a configuation option
            hist_props_defaults = {
                "bins": 100,
                "density": True,
            }

            if hist_props is not None:
                hist_props_defaults.update(hist_props)
            
            H, xbins, ybins = np.histogram2d(_data[:, 0], _data[:, 1], **hist_props_defaults)
            
            xbinmids = xbins[:-1] + (xbins[-1] - xbins[0]) / ((len(xbins) - 1)*2)
            ybinmids = ybins[:-1] + (ybins[-1] - ybins[0]) / ((len(ybins) - 1)*2)

            contour_props_defaults = {
                "cmap": cm.inferno,
            }

            if contour_props is not None:
                contour_props_defaults.update(contour_props)

            if plot == 'contour':
                X, Y = np.meshgrid(xbinmids, ybinmids)
                ax.contour(X, Y, H, **contour_props_defaults)

            if plot == 'contourf':
                X, Y = np.meshgrid(xbinmids, ybinmids)
                ax.contourf(X, Y, H, **contour_props_defaults)

        else:
            raise ValueError(
f"""Plot type {plot} not understood.
Must be one of 'scatter', 'contour'
"""
            )

        ax.set(**ax_props_defaults)
        
        # TODO make this a configuation option
        save_props_defaults = {}

        if save_props is not None:
            save_props_defaults.update(save_props)

        if save:
            fig.savefig(output, **save_props_defaults)
        if show:
            fig.show()


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

    def clean(self, mode='train'):
        if mode == 'train':
            # fixing  missing labels
            n_clusters = len(set(self.train_labels))
            for _cluster in range(1, n_clusters):
                if _cluster not in set(self.train_labels):
                    while _cluster not in set(self.train_labels):
                        self.train_labels[self.train_labels > _cluster] -= 1

            # sorting by clustersize
            n_clusters = np.max(self.train_labels)
            frequency_counts = np.asarray([
                len(np.where(self.train_labels == x)[0]) 
                for x in set(self.train_labels[self.train_labels > 0])
                ])
            old_labels = np.argsort(frequency_counts)[::-1] +1
            proxy_labels = np.copy(self.train_labels)
            for new_label, old_label in enumerate(old_labels, 1):   
                proxy_labels[
                    np.where(self.train_labels == old_label)
                    ] = new_label
            self.train_labels = proxy_labels
            
        else:
            raise NotImplementedError()

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
            self.train_labels = np.zeros(
                np.sum(len(x) for x in self.train_clusterdict.values())
                )
            
            for key, value in self.train_clusterdict.items():
                self.train_labels[value] = key
                """
                if self.train_refindex is None:
                    self.train_labels[value] = key 
                else:
                    self.train_labels[self.train_refindex[value]] = key
                """ 
        else:
            raise NotImplementedError()


    def make_ndx(self, mode='train'):
        if mode == 'train':
            if self.train_clusterdict is None:
                if self.train_labels is not None:
                    self.labels2dict()  
                else:
                    raise LookupError(
                        "No labels or cluster dictionary found for mode 'train'"
                                     )
            part_startpoint = 0
            for count, part in enumerate(range(self.train_shape['parts']), 1):
                part_endpoint = part_startpoint \
                    + self.train_shape['points'][part]
                
                with open(f"rep{count}.ndx", 'w') as file_:
                    for _cluster in self.train_clusterdict.values():
                        _cluster = np.asarray(_cluster)
                        file_.write(f"[ core{_cluster} ]\n")
                        for _member in _cluster[
                            np.where(
                                (_cluster
                                >= part_startpoint)
                                &
                                (_cluster
                                <= part_endpoint))[0]]:
                            
                            file_.write(f"{_member +1}\n")

                part_startpoint = np.copy(part_endpoint)

        else:
            raise NotImplementedError()



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
