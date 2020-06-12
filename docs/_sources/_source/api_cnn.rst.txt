.. _sec_api_cnn:

cnn - A Python module for common-nearest-neighbour (CNN) clustering
===================================================================

Go to:

   * :ref:`CNN class API <sec_api_cnn_CNN>`
   * :ref:`Data input format <sec_api_cnn_input_formats>`
   * :ref:`Cluster results <sec_api_cnn_results>`
   * :ref:`Cluster results <sec_api_cnn_pandas>`
   * :ref:`Cluster results <sec_api_cnn_decorators>`
   * :ref:`Cluster results <sec_api_cnn_functional_api>`

|

.. _sec_api_cnn_CNN:

CNN class API
-------------

The functionality of this module is primarily exposed and bundled by the
:class:`cnnclustering.cnn.CNN` class. For hierarchical clusterings
:class:`cnnclustering.cnn.CNNChild` is used, too.

.. autoclass:: cnnclustering.cnn.CNN
   :members:

|

.. autoclass:: cnnclustering.cnn.CNNChild
   :members:

|

.. _sec_api_cnn_input_formats:

Data input formats
------------------

Input data of differing nature and format (:ref:`points
<sec_api_cnn_points>`, :ref:`distances <sec_api_cnn_distances>`,
:ref:`neighbourhoods <sec_api_cnn_neighbourhoods>`,
:ref:`density graphs <sec_api_cnn_graph>`) are
organised and bundled by the :class:`cnnclustering.cnn.Data` class.

.. autoclass:: cnnclustering.cnn.Data
   :members:

|

.. _sec_api_cnn_points:

Points
^^^^^^

Points are currently supported in the form of a 2D NumPy array of shape
(*n*: points, *d*: dimensions) through the
:class:`cnnclustering.cnn.Points` class.

.. autoclass:: cnnclustering.cnn.Points
   :members:

|

.. _sec_api_cnn_distances:

Distances
^^^^^^^^^

Distances are currently supported in the form of a 2D NumPy array of shape
(*n*: points, *m*: points) through the
:class:`cnnclustering.cnn.Distances` class.

.. autoclass:: cnnclustering.cnn.Distances
   :members:

|

.. _sec_api_cnn_neighbourhoods:

Neighbourhoods
^^^^^^^^^^^^^^

Neighbourhoods are currently supported in the form of:

   * 1D NumPy array of 1D Numpy arrays (
     :class:`cnnclustering.cnn.NeighbourhoodsArray`)
   * Python list of Python sets (
     :class:`cnnclustering.cnn.NeighbourhoodsList`)

Valid neighbourhood containers should inherit from the absract base
class :class:`cnnclustering.cnn.NeighbourhoodsABC` or a the general
realisation :class:`cnnclustering.cnn.Neighbourhoods`.

.. autoclass:: cnnclustering.cnn.NeighbourhoodsArray
   :members:

|

.. autoclass:: cnnclustering.cnn.NeighbourhoodsList
   :members:

|

.. autoclass:: cnnclustering.cnn.Neighbourhoods
   :members:

|

.. autoclass:: cnnclustering.cnn.NeighbourhoodsABC
   :members:

|

.. _sec_api_cnn_graph:

Density graphs
^^^^^^^^^^^^^^

Density graphs are currently supported in the form of a sparse graphs
through the :class:`cnnclustering.cnn.SparsegraphArray` class.

Valid density graph containers should inherit from the absract base
class :class:`cnnclustering.cnn.DensitygraphABC` or a the general
realisation :class:`cnnclustering.cnn.Densitygraph`.

.. autoclass:: cnnclustering.cnn.DensitySparsegraphArray
   :members:

|

.. autoclass:: cnnclustering.cnn.Densitygraph
   :members:

|

.. autoclass:: cnnclustering.cnn.DensitygraphABC
   :members:

|

.. _sec_api_cnn_results:

Cluster results
---------------

Cluster results can be recorded by :meth:`cnnclustering.cnn.CNN.fit` as
:class:`cnnclustering.cnn.CNNRecord`. Multiple records are collected in by
:class:`cnnclustering.cnn.Summary`. Cluster label assignments are stored
independently and are currently supported through
:class:`cnnclustering.cnn.Labels`. Labels can be put into context by
attaching :class:`cnnclustering.cnn.LabelInfo`.

.. autoclass:: cnnclustering.cnn.CNNRecord
   :members:

|

.. autoclass:: cnnclustering.cnn.Summary
   :members:

|

.. autoclass:: cnnclustering.cnn.Labels
   :members:

|

.. autoclass:: cnnclustering.cnn.LabelInfo
   :members:

|


.. _sec_api_cnn_pandas:

Pandas
------

.. autoclass:: cnnclustering.cnn.TypedDataFrame
   :members:

|

.. _sec_api_cnn_decorators:

Decorators
----------

.. autofunction:: cnnclustering.cnn.timed

|

.. autofunction:: cnnclustering.cnn.recorded

|

.. _sec_api_cnn_functional_api:

Functional API
--------------

.. autofunction:: cnnclustering.cnn.calc_dist

|

.. autofunction:: cnnclustering.cnn.fit

|
