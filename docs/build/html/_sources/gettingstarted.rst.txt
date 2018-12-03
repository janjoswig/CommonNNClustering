Getting started
===============

The cnn module provides a data set based interface for 
common-nearest-neighbour clustering. The core functionality is bundled
in a class cnn.CNN(). Here is a minimal example of its usage:

>>> import cnn
...
>>> cobj = cnn.CNN()
>>> cobj.load('path_to_data/data')
>>> cobj.fit()

After importing the main module, this creates an instance of the cnn
cluster class. Data is loaded into the object and fitted, i.e.
clustered by the cnn algorithm. 

Let's go a bit deeper. Call of a cluster objects :py:func:`__str__()`
gives us an overview of its properties.

>>> cobj = cnn.CNN()
>>> print(cobj)
cnn.CNN() cluster object 
alias :                                 root
hierachy level:                         0
test data loaded :                      False
test data shape :                       None
train data loaded :                     False
train data shape :                      None
distance matrix calculated (train):     False
distance matrix calculated (test):      False
clustered :                             False
children :                              False

A freshly created cluster object is by default called *root*. It has the highest possible hierarchy level 0 (more on this later). No data is present as indicated by *data loaded : False* and nothing has been done so far. Data is either treated as *test* or *train* to allow for clustering on one set and interpolation on another. We will see how, but first we need some data: 

>>> from sklearn import datasets
>>> from sklearn.preprocessing import StandardScaler
...
>>> blobs, _ = datasets.make_blobs(n_samples=10000,
...                                cluster_std=[1.0, 2.1, 0.25],
...                                random_state=1)
>>> blobs = StandardScaler().fit_transform(blobs)
>>> cobj = cnn.CNN(test=blobs)

When we now look at our object again, its properties have changed:

>>> print(cobj)
cnn.CNN() cluster object 
alias :                                 root
hierachy level:                         0
test data loaded :                      True
test data shape :                       {'parts': 1, 'points': [10000], 'dimensions': 2}
train data loaded :                     False
train data shape :                      None
distance matrix calculated (train):     False
distance matrix calculated (test):      False
clustered :                             False
children :                              False

Data is now present as *test*. All data is processed as numpy array of
the form (*parts* x *points* x *dimensions*). Data can be passed to the
CNN() class as 1-D (only one part, points in only one dimension) or 2-D
(one part, points in *x* dimensions) array-like structure, but will be
processed internally in this general shape. In the *shape* dictionary, 
``'points'`` is associated with a list of data points per part.

We can get an impression of the loaded data by plotting the points.

>>> cobj.evaluate()

.. figure:: pictures/blobs_reduced.png
   :alt: blobs

Let's reduce the large *test* data set to make the clustering faster.

>>> cobj.cut(points=(None, None, 10))
...
>>> print(cobj)
cnn.CNN() cluster object 
alias :                                 root
hierachy level:                         0
test data loaded :                      True
test data shape :                       {'parts': 1, 'points': [10000], 'dimensions': 2}
train data loaded :                     True
train data shape :                      {'parts': 1, 'points': [1000], 'dimensions': 2}
distance matrix calculated (train):     False
distance matrix calculated (test):      False
clustered :                             False
children :                              False


