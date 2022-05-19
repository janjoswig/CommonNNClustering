List of possible additional features
====================================

Settings mechanism
------------------

Functionality in `cnnclustering.settings` currently not used.

Handling of meta information mappings
-------------------------------------

  - Labels

Provide sanity checks for component combinations
------------------------------------------------

Cleanup `get_subset` methods
----------------------------

Increase test coverage
----------------------

  - NeighboursGetter

DBSCAN clustering
-----------------

Special fitter:

  - Needs to only retrieve one neighbourlist
  - No explicit similarity checker needed

Relaxed CommonNN clustering
---------------------------

Allow check of similarity criterion between all pairs of points removing
the restriction that checked points are neighbours of each other.

Special fitter:

  - Needs to loop over all (unassigned) points for checking instead
    of looping only over members of current neighbour list

Volume scaled CommonNN clustering
---------------------------------

Special similarity checker:

  - Requires not only two neighbour lists and a CommonNN cutoff but
    also pairwise distances

    - Different check signature `Checker.check(input_data, na, nb, c, metric)`
    - Special fitter which uses the specific check signature

  - Restricted to euclidean distances metrics
  - Input data needs to allow retrieval of neighbours and components

Clustering initialisation
-------------------------

Scan for non-sensical keywords that may have been passed (warning)

Children
--------

```python
defaultdict(
  lambda: Clustering(parent=self)
  )
```