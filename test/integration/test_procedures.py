import pytest

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from cnnclustering import cluster
from cnnclustering._types import (
    InputDataExtPointsMemoryview
    )
from cnnclustering._fit import FitterExtBFS


pytestmark = pytest.mark.sklearn

moons_points, moons_reference_labels = datasets.make_moons(
    n_samples=2000, noise=.05
    )
moons_points = StandardScaler().fit_transform(moons_points)


def test_cluster_moons():
    input_data = InputDataExtPointsMemoryview(moons_points)
    neighbours_getter = 
    fitter = FitterExtBFS()
    clustering = cluster.Clustering()