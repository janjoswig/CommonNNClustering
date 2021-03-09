import numpy as np
import pytest

try:
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KDTree
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from cnnclustering import cluster
from cnnclustering._types import (
    InputDataNeighboursSequence,
    InputDataExtPointsMemoryview,
    NeighboursGetterLookup,
    NeighboursGetterExtBruteForce,
    NeighboursList,
    NeighboursExtVector,
    MetricExtEuclidean,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    QueueFIFODeque,
    QueueExtFIFOQueue,
)
from cnnclustering._fit import FitterExtBFS, FitterBFS


pytestmark = pytest.mark.sklearn


def make_moons():
    moons_points, moons_reference_labels = datasets.make_moons(
        n_samples=200, noise=0.05
        )

    moons_points = StandardScaler().fit_transform(moons_points)
    return moons_points, moons_reference_labels


def test_cluster_moons():
    if not SKLEARN_FOUND:
        pytest.skip("Test module requires scikit-learn.")
    # moons_points, moons_reference_labels = make_moons()
    # input_data = InputDataExtPointsMemoryview(moons_points)
    # neighbours_getter = NeighboursGetterExtBruteForce()
    # neighbours = NeighboursExtVector(500)
    # neighbour_neighbours = NeighboursExtVector(500)
    # metric = MetricExtEuclidean()
    # similarity_checker = SimilarityCheckerExtContains()
    # queue = QueueExtFIFOQueue()
    # fitter = FitterExtBFS()

    moons_points, moons_reference_labels = make_moons()
    tree = KDTree(moons_points)
    moons_neighbours = tree.query_radius(
        moons_points, r=0.2, return_distance=False
        )

    input_data = InputDataNeighboursSequence(moons_neighbours)
    neighbours_getter = NeighboursGetterLookup(is_selfcounting=True)
    neighbours = NeighboursList()
    neighbour_neighbours = NeighboursList()
    metric = None
    similarity_checker = SimilarityCheckerContains()
    queue = QueueFIFODeque()
    fitter = FitterBFS()

    clustering = cluster.Clustering(
        input_data=input_data,
        neighbours_getter=neighbours_getter,
        neighbours=neighbours,
        neighbour_neighbours=neighbour_neighbours,
        metric=metric,
        similarity_checker=similarity_checker,
        queue=queue,
        fitter=fitter,
    )

    clustering.fit(0.2, 2)

    np.testing.assert_array_equal(
        clustering._labels,
        moons_reference_labels,
        )
