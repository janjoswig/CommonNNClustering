[![image](https://img.shields.io/pypi/v/cnnclustering.svg)](https://pypi.org/project/cnnclustering/)
[![image](https://img.shields.io/pypi/l/cnnclustering.svg)](https://pypi.org/project/cnnclustering/)
[![image](https://img.shields.io/pypi/pyversions/cnnclustering.svg)](https://pypi.org/project/cnnclustering/)
[![Build Status](https://travis-ci.com/janjoswig/CommonNNClustering.svg?branch=master)](https://travis-ci.com/janjoswig/CommonNNClustering)
[![Coverage Status](https://coveralls.io/repos/github/janjoswig/CommonNNClustering/badge.svg?branch=master)](https://coveralls.io/github/janjoswig/CommonNNClustering?branch=master)

Common-nearest-neighbours clustering
====================================

***
**NOTE**

*This project is currently under development.*
*The implementation may change in the future. Check the examples and the documentation for up-to-date information.*

***

cnnclustering
-------------


The `cnnclustering` Python package provides a flexible interface to use the <b>c</b>ommon-<b>n</b>earest-<b>n</b>eighbours cluster algorithm. While the method can be applied to arbitrary data, this implementation was made before the background of processing trajectories from Molecular Dynamics simulations. In this context the cluster result can serve as a suitable basis for the construction of a core-set Markov-state (cs-MSM) model to capture the essential dynamics of the underlying molecular processes. For a tool for cs-MSM estimation, refer to this separate [project](https://github.com/janjoswig/cs-MSM).

The package provides a main module:

  - `cluster`: (Hierarchical) common-nearest-neighbours clustering and analysis

Features:

  - Flexible: Clustering can be done for data sets in different input formats. Easy interfacing with external methods.
  - Convenient: Integration of functionality, handy in the context of Molecular Dynamics.
  - Fast: Core functionalities implemented in Cython.

Please refer to the following papers for the scientific background (and consider citing if you find the method useful):

  - B. Keller, X. Daura, W. F. van Gunsteren *J. Chem. Phys.*, __2010__, *132*, 074110.
  - O. Lemke, B.G. Keller *J. Chem. Phys.*, __2016__, *145*, 164104.
  - O. Lemke, B.G. Keller *Algorithms*, __2018__, *11*, 19.

Documentation
-------------

The package documentation (under developement) is available [here](https://janjoswig.github.io/CommonNNClustering/).

Install
-------

Refer to the [documentation](https://janjoswig.github.io/CommonNNClustering/_source/install.html) for more details. Install from PyPi

```bash
$ pip install cnnclustering
```

or clone the development version and install from a local branch

```bash
$ git clone https://github.com/janjoswig/CommonNNClustering.git
$ cd CommonNNClustering
$ pip install .
```

Quickstart
----------

```python
>>> from cnnclustering.cluster import prepare_clustering

>>> # 2D data points (list of lists, 12 points in 2 dimensions)
>>> data_points = [   # point index
...     [0, 0],       # 0
...     [1, 1],       # 1
...     [1, 0],       # 2
...     [0, -1],      # 3
...     [0.5, -0.5],  # 4
...     [2,  1.5],    # 5
...     [2.5, -0.5],  # 6
...     [4, 2],       # 7
...     [4.5, 2.5],   # 8
...     [5, -1],      # 9
...     [5.5, -0.5],  # 10
...     [5.5, -1.5],  # 11
...     ]

>>> clustering = prepare_clustering(data_points)
>>> clustering.fit(radius_cutoff=1.5, cnn_cutoff=1, v=False)
>>> clustering.labels
Labels([1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2])

```

![quickstart](docs/_images/tutorial_basic_usage_27_0.png)


Alternative scikit-learn implementation
---------------------------------------

We provide an alternative approach to common-nearest-neighbours clustering in the spirit of the scikit-learn project within [scikit-learn-extra](https://github.com/scikit-learn-contrib/scikit-learn-extra).
