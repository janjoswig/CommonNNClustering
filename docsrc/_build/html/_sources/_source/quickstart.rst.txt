Quickstart
==========

.. code-block::

   from cnnclustering import cnn
   import matplotlib.pyplot as plt

   # 2D data points (list of lists, 12 points in 2 dimensions)
   data_points = [   # point index
      [0, 0],       # 0
      [1, 1],       # 1
      [1, 0],       # 2
      [0, -1],      # 3
      [0.5, -0.5],  # 4
      [2,  1.5],    # 5
      [2.5, -0.5],  # 6
      [4, 2],       # 7
      [4.5, 2.5],   # 8
      [5, -1],      # 9
      [5.5, -0.5],  # 10
      [5.5, -1.5],  # 11
      ]

   clustering = cnn.CNN(points=data_points)
   clustering.fit(radius_cutoff=1.5, cnn_cutoff=1)
   clustering.labels
   # Labels([1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2])

   fig, ax = plt.subplots(1, 2)
   ax[0].set_title("original")
   clustering.evaluate(ax=ax[0], original=True)

   ax[1].set_title("clustered")
   clustering.evaluate(ax=ax[1])

.. image:: ../_build/html/_images/tutorial_basic_usage_42_0.png