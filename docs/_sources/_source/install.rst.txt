Installation instructions
=========================

Requirements
------------

The :mod:`cnnclustering` package is developed and tested in Python >=3.6.
At runtime the package has a few mandatory third party dependencies.
We recommend to install the latest version of:

   * :mod:`numpy`
   * :mod:`PyYAML`
   * :mod:`tqdm`

Optionally, additional functionality is available when the following
packages are installed as well:

   * :mod:`matplotlib`
   * :mod:`pandas`
   * :mod:`networkx`
   * :mod:`scipy`
   * :mod:`sklearn`

PyPi
----

.. code-block:: bash

   pip install cnnclustering

Developement installation
-------------------------

Clone the source repository `https://github.com/janjoswig/CommonNNClustering
<https://github.com/janjoswig/CommonNNClustering>`_ and use the package
:mod:`cnnclustering` as you prefer it, e.g. install it via pip in editable mode.

.. code-block:: bash

   $ git clone https://github.com/janjoswig/CommonNNClustering
   $ cd CommonNNClustering
   $ pip install -e .

To recompile the Cython-extensions (requires :mod:`cython` installed) use:

.. code-block:: bash

   $ python setup.py build_ext --inplace
