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
    """Decorator to measure execution time"""
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
    """Decorator to format function feedback"""
    @wraps(function_)
    def wrapper(self, *args, **kwargs):
        wrapped = function_(self, *args, **kwargs)
        if wrapped is not None:
            wrapped[-2]['time'] = wrapped[-1]
            print(f'recording: ... \n{wrapped[-2]}')
            self.summary = self.summary.append(wrapped[-2], ignore_index=True)
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

    def __init__(self, alias='root', data=None, reduced=None, dist_matrix=None,
                 map_matrix=None):
        self.alias = alias
        self.hierarchy_level = 0
        self.data = data
        self.rdata = reduced
        self.dist_matrix = dist_matrix
        self.map_matrix = map_matrix
        self.data, self.shape = self.get_shape(self.data)
        self.rdata, self.rshape = self.get_shape(self.rdata)
        self.clusters = None
        self.labels = None
        self.summary = pd.DataFrame(columns=record._fields)
        self.children = None
    
    def check(self):
        if self.data is not None:
            self.data_present = True
        else:
            self.data_present = False 

        if self.rdata is not None:
            self.rdata_present = True
        else:
            self.rdata_present = False
        
        if self.dist_matrix is not None:
            self.dist_matrix_present = True
        else:
            self.dist_matrix_present = False

        if self.clusters is not None:
            self.clusters_present = True
        else:
            self.clusters_present = False
        
        if self.children is not None:
            self.children_present = True
        else:
            self.children_present = False

    def __str__(self):
        self.check()
        return f"""cnn.CNN() cluster object 
alias :                          {self.alias}
hierachy level:                  {self.hierarchy_level}
data loaded :                    {self.data_present}
shape :                          {self.shape}
reduced :                        {self.rdata_present}
reduced shape :                  {self.rshape}
distance matrix calculated :     {self.dist_matrix_present}
clustered :                      {self.clusters_present}
children:                        {self.children_present}
"""

    def load(self, file_, output=None):
        """Loads file content and return data and shape"""
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
        
        if output is None:
            self.data, self.shape = self.get_shape(data)
        else:
            return self.get_shape(data)
    
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

    def reduce(self, data=None, parts=(None, None, None), points=(None, None, None),
               dimensions=(None, None, None)):
        """Alows data set reduction.  For each data set level (parts,
        points, dimensions) a tuple (start:stop:step) can be
        specified."""
        if data is None:
            self.rdata = [x[slice(*points), slice(*dimensions)] 
                          for x in self.data[slice(*parts)]]
        
            self.rdata, self.rshape = self.get_shape(self.rdata)
        else:
            data, _ = self.get_shape(data)
            return [x[slice(*points), slice(*dimensions)] 
                    for x in data[slice(*parts)]]

    @timed
    def dist(self, raw=False, low_memory=False):
        """Computes a distance matrix points x points for points in given data
        of standard shape (parts, points, dimensions)"""

        data, _ = self.query_data(raw)
        points = np.vstack(data)       

        print(len(points))

        if low_memory:
            raise NotImplementedError()
        else:
            self.dist_matrix = cdist(points, points)

    @timed
    def map(self, nearest=None):
        """Computes a map matrix that maps an arbitrary data set to a
        reduced to set"""

        if self.data or self.rdata is None:
            raise LookupError(
                "Mapping requires an original and a reduced data set"
                )
        elif self.shape['dimensions'] != self.rshape['dimensions']:
            raise ValueError(
                "Mapping requires the same number of dimension in the original \
                 and the reduced data set"
                )

        if nearest is not None:
            raise NotImplementedError()
        else:
            self.map_matrix = cdist(np.vstack(self.data), np.vstack(self.rdata))
            
    def dist_hist(self, dist_matrix=None, bins=200, range=None,
                  density=True, weights=None, xlabel='d / nm', ylabel='',
                  show=True, save=False):
        """Shows/saves a histogram plot for distances in a given distance
        matrix"""
        if dist_matrix is None:
            dist_matrix = self.dist_matrix
        flat_ = np.tril(dist_matrix).flatten()
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
        plt.tight_layout(pad=0.1)
        if show:
            plt.show()
        else:
            pass
        plt.close()
    
    @recorded
    @timed
    def cluster(self, dist_matrix=None, radius_cutoff=1, cnn_cutoff=1,
                member_cutoff=1, max_clusters=None, output=None):
        """Performs a CNN clustering of points in a given distance matrix"""
        if dist_matrix is None:
            if self.dist_matrix is None:
                self.dist()
            _dist_matrix = self.dist_matrix
            # what if dist_matrix not available -> low memory
        else:
            _dist_matrix = dist_matrix

        n_points = len(_dist_matrix)
        neighbours = np.asarray([
            np.where(x <= radius_cutoff)[0] for x in _dist_matrix
            ])
        n_neighbours = np.asarray([len(x) for x in neighbours])
        include = np.ones(len(neighbours), dtype=bool)
        include[np.where(n_neighbours < cnn_cutoff)[0] ] = False
        
        clusters = defaultdict(list)
        clusters[0].extend(np.where(include == False)[0])
        labels = np.zeros(n_points).astype(int)
        current = 1

        enough = False
        while any(include) and not enough:
            point = np.where(
                (n_neighbours == np.max(n_neighbours[include]))
                & (include == True)
            )[0][0]
            clusters[current].append(point)
            new_point_added = True
            labels[point] = current
            include[point] = False

            done = 0
            while new_point_added:
                new_point_added = False
                for member in clusters[current][done:]:
                    for neighbour in neighbours[member]:
                        if include[neighbour]:
                            common_neighbours = (
                                set(neighbours[member])
                                & set(neighbours[neighbour])
                                )

                            if len(common_neighbours) >= cnn_cutoff:
                            # and (point in neighbours[neighbour])
                            # and (neighbour in neighbours[point]):
                                clusters[current].append(neighbour)
                                new_point_added = True
                                labels[neighbour] = current
                                include[neighbour] = False

                done += 1   
            current += 1

            if max_clusters is not None:
                if current == max_clusters+1:
                    enough = True
        
        clusters_no_noise = {y: clusters[y] 
                    for y in clusters if y != 0}
        
        too_small = [
            clusters.pop(y) 
            for y in [x[0] 
            for x in clusters_no_noise.items() if len(x[1]) < member_cutoff]
            ]
        
        if len(too_small) > 0:
            clusters[0].extend(too_small)
        
        for x in set(labels):
            if x not in set(clusters):
                labels[labels == x] = 0

        if len(clusters_no_noise) == 0:
            largest = 0
        else:
            largest = len(clusters[1 + np.argmax([
                len(x) 
                for x in clusters_no_noise.values()
                    ])]) / n_points
        
        recording = pd.Series([
                        n_points,
                        radius_cutoff,
                        cnn_cutoff,
                        member_cutoff,
                        max_clusters,
                        len(clusters) -1,
                        largest,
                        len(clusters[0]) / n_points,
                        None,
                        ],
                        index=record._fields,
                        dtype='object',
                        )

        if dist_matrix is None:
            self.clusters = clusters
            self.labels = labels
        
            return recording
        
        else:
            return (clusters, labels), recording

    def predict(self, data=None, rdata=None, rlabels=None, map_matrix=None,
                low_memory=False, radius_cutoff=1, cnn_cutoff=1, member_cutoff=1,
                max_clusters=None):

        if low_memory:
            _data, _shape = self.query_data(self, data, default='data')
            _rdata, _rshape = self.query_data(self, data, default='data')
        else:
            _map_matrix = np.vstack(self.map_matrix)
            _dist_matrix = self.dist_matrix
            rlabels = self.labels

            neighbours = np.asarray([
                np.where(x <= radius_cutoff)[0] for x in _map_matrix
                ])
            rneighbours = np.asarray([                  
                np.where(x <= radius_cutoff)[0] for x in _dist_matrix
                ])    

            n_points = len(_map_matrix)
            labels = np.zeros(n_points).astype(int)
            for point in range(n_points):
                print(f'Predicting cluster for point {1+point:6} of {n_points}',
                    end='\r')
                same = np.where(_map_matrix[point] == 0)[0]
                if len(same) > 0:
                    labels[point] = rlabels[same[0]]
                else:
                    common_neighbours = [
                        set(neighbours[point])
                        & set(rneighbours[x])
                        for x in neighbours[point]
                        ]

                    cnn_fulfilled = np.where(
                        np.asarray([
                            len(x) for x in common_neighbours
                            ]) >= cnn_cutoff)[0]
              
                    if len(cnn_fulfilled) > 0:
                        labels[point] = rlabels[
                            neighbours[point][
                                np.argmax(cnn_fulfilled)]]                   

        return labels


    def query_data(self, raw):
        """Helper function to evaluate user input. If data is required as
        keyword argument and data=None is passed, the default data used is
        either self.rdata or self.data."""
        if not raw:
            if self.rdata is not None:
                _data = self.rdata
                _shape = self.rshape
            elif self.data is None:
                raise LookupError(
                    "No data available"
                    )
            else:
                _data = self.data
                _shape = self.shape  
        else:
            if self.data is not None:
                _data = self.data
                _shape = self.shape
            elif self.rdata is None:
                raise LookupError(
                    "No data available"
                    )
            else:
                _data = self.rdata
                _shape = self.rshape        

        return _data, _shape


    def evaluate(self, raw=False, labels=None, data=None, max_clusters=None,
                 mode='scatter', dim=None):
        """Shows/saves a 2D histogram or scatter plot of a cluster result"""

        _data, _ = self.query_data(self, data)
        if dim is None:
            dim = 0
        _data = self.reduce(_data, dimensions=(dim, dim+2, 1))
        _data = np.vstack(_data)

        if (raw or self.labels is None) and labels is None:
            _labels = np.ones(len(_data)).astype(int)
        elif labels is None:
            _labels = self.labels
        else:
            _labels = labels
       
        if max_clusters is not None:
            _labels[_labels > max_clusters] = 0

        colors = np.array(list(islice(cycle(['#000000', '#396ab1', '#da7c30',
                                             '#3e9651', '#cc2529', '#535154',
                                             '#6b4c9a', '#922428', '#948b3d']),
                                int(max(_labels) + 1))))

        fig, ax = plt.subplots()
        ax.scatter(_data[:, 0], _data[:, 1], s=10, color=colors[_labels])
        plt.show()
        plt.close()


    def isolate(self, raw=False, clusters=None, purge=True):
        """Isolates points per clusters based on a cluster result"""
        if purge or self.children is None:
            self.children = defaultdict(lambda: CNNChild(self))
        
        _data, _shape = self.query_data(raw)
        #_data = self.data
        #_shape = self.shape

        if clusters is None:
            clusters = self.clusters

        for cluster in clusters:
            if len(clusters[cluster]) > 0:
                cluster_data = []
                part_startpoint = 0
                for part in range(_shape['parts']):
                    part_endpoint = part_startpoint \
                        + _shape['points'][part] -1
                    sorted_members = np.asarray(sorted(clusters[cluster]))
                    cluster_data.append(
                        _data[part][sorted_members[
                            np.where(
                                (sorted_members
                                >= part_startpoint)
                                &
                                (sorted_members
                                <= part_endpoint))[0]]]
                            )
                    part_startpoint = np.copy(part_endpoint)

                self.children[cluster].alias = f'child No. {cluster}'
                self.children[cluster].data = cluster_data
                _, self.children[cluster].shape = \
                self.children[cluster].get_shape(
                    self.children[cluster].data
                    )

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
