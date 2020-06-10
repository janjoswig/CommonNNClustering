Common nearest neighbours (CNN) clustering and core-set Markov-state model estimation 
=====================================================================================

***
**NOTE**

*This project is currently under development in the alpha state.*
*The implementation may change in the future. Check the examples and the documentation for up-to-date information.*

***

cnnclustering
-------------


The `cnnclustering` Python package provides a flexible interface to use the <b>c</b>ommon-<b>n</b>earest-<b>n</b>eighbours cluster algorithm. While the method can be applied to abitrary data, this implementation was made before the background of processing trajectories from Molecular Dynamics simulations. In this context the cluster result can serve as a suitable basis for the construction of a core-set Markov-state (csMSM) model to capture the essential dynamics of the underlying molecular processes. 

The package provides two modules:

  - `cnn`: (Hierarchical) CNN clustering and analysis
  - `cmsm`: csMSM estimation and analysis
   
Features:

  - Flexible: Clustering can be done for data sets in different input formats. Easy interfacing with external methods.
  - Convenient: Integration of functionality, handy in the context of Molecular Dynamics.
  - Fast: Core functionalities use Cython.
  
Please refer to the following papers for the scientific background (and consider citing if you find the method useful):

  - B. Keller, X. Daura, W. F. van Gunsteren *J. Chem. Phys.*, __2010__, *132*, 074110.
  - O. Lemke, B.G. Keller, *J. Chem. Phys.*, __2016__, *145*, 164104.
  - O. Lemke, B.G. Keller, *Algorithms*, __2018__, *11*, 19.

Documentation
-------------

The package documentation (under developement) is available [here](https://janjoswig.userpage.fu-berlin.de).

Quickstart
----------

Alternative scikit-learn implementation
---------------------------------------

We provide an alternative approach to CNN clustering in the spirit of the scikit-learn project over this [fork](https://github.com/janjoswig/scikit-learn/tree/cnnclustering).
