Quickstart
==========

.. code-block::

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

.. image:: ../_build/html/_images/tutorial_basic_usage_27_0.png