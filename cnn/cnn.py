"""This is cnn v0.1

The functionality provided in this module is based on code implemented
by Oliver Lemke in the script collection CNNClustering available on
git-hub (clone https://github.com/BDGSoftware/CNNClustering.git).

Author: Jan Joswig, 
first edit: 09.10.18
released:
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
from sortedcontainers import SortedList
from scipy.spatial.distance import cdist
from functools import wraps
import time
import pandas as pd
from itertools import cycle, islice

################################################################################

def timed(function_):
    """Decorator to measure execution time.  Forwards the output of the
       wrapped function and measured excecution time."""
    @wraps(function_)
    def wrapper(*args, **kwargs):
        go = time.time()
        wrapped = function_(*args, **kwargs)
        stop = time.time()
        
        stopped = stop - go
        hours, rest = divmod(stopped, 3600)
        minutes, seconds = divmod(rest, 60)
        print(
f"Execution time for call of {function_.__name__}(): \
{int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds")
        return wrapped, stopped
    return wrapper

# generic function feedback data container for CCN.cluster(); used only
# to provide column identifiers.  maybe not too useful ...
record = namedtuple('clusterRecord',
                    ['points',
                    'radius_cutoff',
                    'cnn_cutoff',
                    'member_cutoff',
                    'max_clusters',
                    'n_cluster',
                    'largest',
                    'noise',
                    'time'])

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
                self.summary = self.summary.append(wrapped, ignore_index=True)
        return
    return wrapper

class CNN():
    """CNN cluster object class"""
    
    def get_shape(self, data):
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

    def __init__(self, alias='root', train=None, test=None, dist_matrix=None,
                 map_matrix=None):
        self.alias = alias
        self.hierarchy_level = 0
        self.test = test
        self.train = train
        self.dist_matrix = dist_matrix
        self.map_matrix = map_matrix
        self.test, self.test_shape = self.get_shape(self.test)
        self.train, self.train_shape = self.get_shape(self.train)
        self.test_clusterdict = None
        self.test_labels = None
        self.train_clusterdict = None
        self.train_labels = None
        self.summary = pd.DataFrame(columns=record._fields)
        self.train_children = None
        self.train_refindex = None


    def check(self):
        if self.test is not None:
            self.test_present = True
        else:
            self.test_present = False 

        if self.train is not None:
            self.train_present = True
        else:
            self.train_present = False
        
        if self.dist_matrix is not None:
            self.dist_matrix_present = True
        else:
            self.dist_matrix_present = False

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
alias :                          {self.alias}
hierachy level:                  {self.hierarchy_level}
test data loaded :               {self.test_present}
test data shape :                {self.test_shape}
train data loaded :              {self.train_present}
train data shape :               {self.train_shape}
distance matrix calculated :     {self.dist_matrix_present}
clustered :                      {self.clusters_present}
children :                       {self.children_present}
"""

    def load(self, file_, mode='train'):
        """Loads file content and return data and shape"""
        # add load option for dist_matrix, map_matrix
        
        extension = file_.rsplit('.', 1)[-1]
        if len(extension) == 1:
            extension = ''
        case_ = {
            'p' : lambda: pickle.load(open(file_, 'rb')),
            'npy': lambda: np.load(file_),
             '': lambda: np.loadtxt(file_),
             }
        data = case_.get(extension,
            f"Unknown filename extension .{extension}")()
    
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
        extension = file_.rsplit('.', 1)[1]
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
        elif self.test is None:
            raise LookupError(
                "Neither test nor train data present."
                )

        self.train = [x[slice(*points), slice(*dimensions)] 
                        for x in self.test[slice(*parts)]]
    
        self.train, self.train_shape = self.get_shape(self.train)

    @timed
    def dist(self, low_memory=False):
        """Computes a distance matrix points x points for points in given data
        of standard shape (parts, points, dimensions)"""

        if (self.train is None) and (self.test is not None):
            print(
                "No train data present, but test data found. Switching data."    
                 )
            self.switch_data()

        points = np.vstack(self.train)       

        print(
            f"Calculating nxn distance matrix for {len(points)} points"
             )
        if low_memory:
            raise NotImplementedError()
        else:
            self.dist_matrix = cdist(points, points)

    @timed
    def map(self, nearest=None):
        """Computes a map matrix that maps an arbitrary data set to a
        reduced to set"""

        if self.train or self.test is None:
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
            self.map_matrix = cdist(np.vstack(self.test), np.vstack(self.train))
            
    def dist_hist(self, bins=200, range=None,
                  density=True, weights=None, xlabel='d / au', ylabel='',
                  show=True, save=False, output='dist_hist.pdf', dpi=300):
        """Shows/saves a histogram plot for distances in a given distance
        matrix"""
        if self.dist_matrix is None:
            print(
                "Distance matrix not calculated. Calculating distance matrix."
                 )
            self.dist()

        flat_ = np.tril(self.dist_matrix).flatten()
        histogram, bins =  np.histogram(flat_[flat_ > 0],
                                        bins=bins,
                                        range=range,
                                        density=density,
                                        weights=weights)
        binmids = bins[:-1] + (bins[-1] - bins[0]) / ((len(bins) - 1)*2)                                                                   
        fig, ax = plt.subplots()
        ax.plot(binmids, histogram)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, np.max(binmids))
        plt.tight_layout(pad=0.1)
        if save:
            plt.savefig(output, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    
    @recorded
    @timed
    def cluster(self, radius_cutoff=1, cnn_cutoff=1,
                member_cutoff=1, max_clusters=None, rec=True):
        """Performs a CNN clustering of points in a given distance matrix"""
        if self.dist_matrix is None:
            self.dist()
        
        n_points = len(self.dist_matrix)
        neighbours = np.asarray([
            np.where(x <= radius_cutoff)[0] for x in self.dist_matrix
            ])
        n_neighbours = np.asarray([len(x) for x in neighbours])
        include = np.ones(len(neighbours), dtype=bool)
        include[np.where(n_neighbours < cnn_cutoff)[0] ] = False
        
        _clusterdict = defaultdict(list)
        _clusterdict[0].extend(np.where(include == False)[0])
        _labels = np.zeros(n_points).astype(int)
        current = 1

        enough = False
        while any(include) and not enough:
            point = np.where(
                (n_neighbours == np.max(n_neighbours[include]))
                & (include == True)
            )[0][0]
            _clusterdict[current].append(point)
            new_point_added = True
            _labels[point] = current
            include[point] = False

            done = 0
            while new_point_added:
                new_point_added = False
                for member in _clusterdict[current][done:]:
                    for neighbour in neighbours[member]:
                        if include[neighbour]:
                            common_neighbours = (
                                set(neighbours[member])
                                & set(neighbours[neighbour])
                                )

                            if len(common_neighbours) >= cnn_cutoff:
                            # and (point in neighbours[neighbour])
                            # and (neighbour in neighbours[point]):
                                _clusterdict[current].append(neighbour)
                                new_point_added = True
                                _labels[neighbour] = current
                                include[neighbour] = False

                done += 1   
            current += 1

            if max_clusters is not None:
                if current == max_clusters+1:
                    enough = True
        
        clusters_no_noise = {y: _clusterdict[y] 
                    for y in _clusterdict if y != 0}
        
        too_small = [
            _clusterdict.pop(y) 
            for y in [x[0] 
            for x in clusters_no_noise.items() if len(x[1]) < member_cutoff]
            ]
        
        if len(too_small) > 0:
            _clusterdict[0].extend(too_small)
        
        for x in set(_labels):
            if x not in set(_clusterdict):
                _labels[_labels == x] = 0

        if len(clusters_no_noise) == 0:
            largest = 0
        else:
            largest = len(_clusterdict[1 + np.argmax([
                len(x) 
                for x in clusters_no_noise.values()
                    ])]) / n_points

        self.train_clusterdict = _clusterdict
        self.train_labels = _labels

        if rec:
            return pd.Series([
                        n_points,
                        radius_cutoff,
                        cnn_cutoff,
                        member_cutoff,
                        max_clusters,
                        len(_clusterdict) -1,
                        largest,
                        len(_clusterdict[0]) / n_points,
                        None,
                        ],
                        index=record._fields,
                        dtype='object',
                        )

    def predict(self, low_memory=False, radius_cutoff=1, cnn_cutoff=1,
                member_cutoff=1, max_clusters=None, cluster=None):

        if low_memory:
            raise NotImplementedError()
        else:
            if self.map_matrix is None:
                self.map()

            test_neighbours = np.asarray([
                np.where(x <= radius_cutoff)[0] for x in self.map_matrix
                ])
            train_neighbours = np.asarray([                  
                np.where(x <= radius_cutoff)[0] for x in self.dist_matrix
                ])    

            n_points = len(self.map_matrix)
            test_labels = np.zeros(n_points).astype(int)
            for point in range(n_points):
                print(f'Predicting cluster for point {1+point:6} of {n_points}',
                    end='\r')
                same = np.where(self.map_matrix[point] == 0)[0]
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

        self.test_labels = test_labels


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


    def evaluate(self, mode='train', max_clusters=None,
                 plot='scatter', dim=None, show=True, save=False,
                 output='evaluation.pdf', dpi=300):
        """Shows/saves a 2D histogram or scatter plot of a cluster result"""

        _data, _ = self.query_data(mode=mode)
        if dim is None:
            dim = 0
        _data = [x[slice(None, None, None), slice(dim, dim+2, 1)] 
                        for x in _data[slice(None, None, None)]]

        _data = np.vstack(_data)

        if mode == 'test':
            if self.test_labels is None:
                _labels = np.ones(len(_data)).astype(int)
            else:
                _labels = self.test_labels
        elif mode == 'train':
            if self.train_labels is None:
                _labels = np.ones(len(_data)).astype(int)
            else:
                _labels = self.train_labels
        
       
        if max_clusters is not None:
            _labels[_labels > max_clusters] = 0

        colors = np.array(list(islice(cycle(['#000000', '#396ab1', '#da7c30',
                                             '#3e9651', '#cc2529', '#535154',
                                             '#6b4c9a', '#922428', '#948b3d']),
                                int(max(_labels) + 1))))

        fig, ax = plt.subplots()
        ax.scatter(_data[:, 0], _data[:, 1], s=10, color=colors[_labels])
        if save:
            plt.savefig(output, dpi=dpi)
        if show:
            plt.show()
        plt.close()


    def isolate(self, mode='train', purge=True):
        """Isolates points per clusters based on a cluster result"""        
        
        if mode == 'train':
            if purge or self.train_children is None:
                self.train_children = defaultdict(lambda: CNNChild(self))
            
            for _cluster in  self.train_clusterdict:
                if len(self.train_clusterdict[_cluster]) > 0:
                    ref_index = []
                    cluster_data = []
                    part_startpoint = 0
                    for part in range(self.train_shape['parts']):
                        part_endpoint = part_startpoint \
                            + self.train_shape['points'][part] -1
                        sorted_members = np.asarray(
                            sorted(self.train_clusterdict[_cluster])
                            )
                        if self.train_refindex is None:
                            ref_index.extend(sorted_members)
                        else:
                            ref_index.extend(
                                self.train_refindex[sorted_members]
                                )

                        cluster_data.append(
                            self.train[part][sorted_members[
                                np.where(
                                    (sorted_members
                                    >= part_startpoint)
                                    &
                                    (sorted_members
                                    <= part_endpoint))[0]]]
                                )
                        part_startpoint = np.copy(part_endpoint)

                    self.train_children[_cluster].alias = f'child No. {_cluster}'
                    self.train_children[_cluster].train, \
                    self.train_children[_cluster].train_shape = \
                    self.train_children[_cluster].get_shape(cluster_data)
                    self.train_children[_cluster].train_refindex = np.asarray(
                                                                       ref_index
                                                                       )
        else:
            raise NotImplementedError()


    def reel(self):
        if self.train_children is None:
            raise LookupError(
                "No child clusters isolated"
                             )

        for _cluster in self.train_children.values():
            n_clusters = max(self.train_clusterdict)
            if _cluster.train_labels is not None:
                self.train_labels[
                _cluster.train_refindex[
                    np.where(_cluster.train_labels == 0)[0]
                    ]
                ] = 0

                for _label in _cluster.train_labels[_cluster.train_labels > 1]:
                    self.train_labels[
                    _cluster.train_refindex[
                        np.where(_cluster.train_labels == _label)[0]
                        ]
                    ] = _label + n_clusters                   

    def clean(self, mode='train'):
        if mode == 'train':
            # fixing  missing labels
            n_clusters = np.max(self.train_labels)
            for _cluster in range(1, n_clusters +1):
                if _cluster not in self.train_labels:
                    self.train_labels[self.train_labels > _cluster] -= 1

            # sorting by clustersize
            n_clusters = np.max(self.train_labels)
            frequency_counts = [
                len(np.where(self.train_labels == x)[0]) 
                for x in set(self.train_labels[self.train_labels > 0])
                ]
            new_labels = n_clusters - np.argsort(frequency_counts)
            proxy_labels = np.copy(self.train_labels)
            for old_label, new_label in enumerate(new_labels, 1):   
                proxy_labels[
                    np.where(self.train_labels == old_label)
                    ] = new_label
            self.train_labels = proxy_labels

            self.clean()
            self.labels2dict()
            
        else:
            raise NotImplementedError()

    def labels2dict(self, mode='train'):
        if mode == 'train':
            self.train_clusterdict = defaultdict(list)
            for _cluster in range(np.max(self.train_labels) +1):
                if self.train_refindex is None:
                    self.train_clusterdict[_cluster].extend(
                        np.where(self.train_labels == _cluster)[0]
                        )
                else:
                    self.train_clusterdict[_cluster].extend(
                        self.train_refindex[
                        np.where(self.train_labels == _cluster)[0]
                        ])                 
        else:
            raise NotImplementedError()

    def dict2labels(self, mode='train'):
        if mode == 'train':
            self.train_labels = np.zeros(
                np.sum(len(x) for x in self.train_clusterdict.values())
                )
            
            for key, value in self.train_clusterdict.items():
                if self.train_refindex is None:
                    self.train_labels[value] = key 
                else:
                    self.train_labels[self.train_refindex[value]] = key 
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
            for part in range(1, self.train_shape['parts']+1):
                part_endpoint = part_startpoint \
                    + self.train_shape['points'][part] -1
                
                with open(f"rep{part}.ndx", 'w') as file_:
                    for _cluster in self.train_clusterdict:
                        sorted_members = np.asarray(
                            sorted(self.train_clusterdict[_cluster])
                            )
                        file_.write(f"[ core{_cluster} ]\n")
                        for _member in sorted_members[
                            np.where(
                                (sorted_members
                                >= part_startpoint)
                                &
                                (sorted_members
                                <= part_endpoint))[0]]:
                            
                            file_.write(f"{_member +1}\n")

                part_startpoint = np.copy(part_endpoint)

        else:
            raise NotImplementedError()

class CNNChild(CNN):
    def __init__(self, parent):
        super().__init__()
        self.hierarchy_level = parent.hierarchy_level +1
        self.alias = 'child'

########################################################################

def dist(data):
    cobj = CNN(data)
    cobj.dist()
    return cobj.dist_matrix

########################################################################

if __name__ == "__main__":
    print(__name__)
