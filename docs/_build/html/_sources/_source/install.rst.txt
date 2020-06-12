Installation instructions
=========================

Requirements
------------

The :mod:`cnnclustering` package is developed and tested in Python 3.8. Earlier
versions >=3.6 may work as well, though. The package has a few third
party Python package dependencies. We recommend to install the latest
version of:

   * :mod:`matplotlib`
   * :mod:`numpy`
   * :mod:`scipy`
   * :mod:`pandas` (optional)
   * :mod:`tqdm`
   * :mod:`PyYAML`

If you want to compile the Cython extensions yourself, you will need to
install :mod:`Cython` as well. The tests can be run after installing
:mod:`pytest` and require :mod:`scikit-learn`.

PyPi (commings soon)
--------------------

.. code-block:: bash

   pip install cnnclustering

Manual installation
-------------------

Clone the source repository `https://github.com/janjoswig/CNN
<https://github.com/janjoswig/CNN>`_ and use the package
:mod:`cnnclustering` as you prefer it, e.g. install it via pip.

.. code-block:: bash

   $ git clone https://github.com/janjoswig/CNN
   $ cd CNN
   $ pip install .

