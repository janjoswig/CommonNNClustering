import pytest

try:
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from cnnclustering import cluster
from cnnclustering._types import (
    InputDataExtPointsMemoryview,
    NeighboursGetterExtBruteForce,
    NeighboursExtVector,
    MetricExtEuclidean,
    SimilarityCheckerExtContains,
    QueueExtFIFOQueue,
)
from cnnclustering._fit import FitterExtBFS


pytestmark = pytest.mark.sklearn


def make_moons():
    moons_points, moons_reference_labels = datasets.make_moons(
        n_samples=2000, noise=0.05
        )

    moons_points = StandardScaler().fit_transform(moons_points)
    return moons_points, moons_reference_labels


@pytest.mark.skip("debug")
def test_cluster_moons():
    if not SKLEARN_FOUND:
        pytest.skip("Test module requires scikit-learn.")
    moons_points, moon_reference_labels = make_moons()
    input_data = InputDataExtPointsMemoryview(moons_points)
    neighbours_getter = NeighboursGetterExtBruteForce()
    neighbours = NeighboursExtVector(500)
    neighbour_neighbours = NeighboursExtVector(500)
    metric = MetricExtEuclidean()
    similarity_checker = SimilarityCheckerExtContains()
    queue = QueueExtFIFOQueue()
    fitter = FitterExtBFS()
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

    clustering.fit(0.2, 5)

    print(moons_reference_labels)
    print(clustering._labels)
